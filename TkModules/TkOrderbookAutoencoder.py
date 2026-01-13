import configparser
import torch
import json
from TkModules.TkModel import TkModel

#------------------------------------------------------------------------------------------------------------------------

class VectorQuantizerEMA(torch.nn.Module):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        commitment_cost=0.25,
        decay=0.99,
        eps=1e-5,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.eps = eps

        self.embedding = torch.nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.normal_()

        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_w", self.embedding.weight.data.clone())

    def quantize(self, z):
        # z: (B, C, T)
        z = z.permute(0, 2, 1).contiguous()  # (B, T, C)
        flat_z = z.view(-1, self.embedding_dim)  # (B*T, C)

        distances = (
            flat_z.pow(2).sum(1, keepdim=True)
            - 2 * flat_z @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(1)
        )

        encoding_indices = torch.argmin(distances, dim=1)
        encodings = torch.nn.functional.one_hot(encoding_indices, self.num_embeddings).type(flat_z.dtype)

        quantized = self.embedding(encoding_indices).view(z.shape)

        # EMA updates (training only)
        if self.training:
            self.ema_cluster_size.mul_(self.decay).add_(
                encodings.sum(0), alpha=1 - self.decay
            )

            dw = encodings.t() @ flat_z
            self.ema_w.mul_(self.decay).add_(dw, alpha=1 - self.decay)

            n = self.ema_cluster_size.sum()
            cluster_size = (
                (self.ema_cluster_size + self.eps)
                / (n + self.num_embeddings * self.eps)
                * n
            )

            self.embedding.weight.data.copy_(self.ema_w / cluster_size.unsqueeze(1))

        # commitment loss
        loss = self.commitment_cost * torch.nn.functional.mse_loss(quantized.detach(), z)

        # straight-through estimator
        quantized = z + (quantized - z).detach()
        quantized = quantized.permute(0, 2, 1).contiguous()

        codes = encoding_indices.view(z.shape[0], z.shape[1])  # (B, T)

        return quantized, loss, codes

    def forward(self, z):
        z_q, loss, codes = self.quantize(z)
        return z_q, loss, codes

#------------------------------------------------------------------------------------------------------------------------

class TkOrderbookAutoencoder(torch.nn.Module):

    def __init__(self, _cfg : configparser.ConfigParser):

        super(TkOrderbookAutoencoder, self).__init__()

        self._cfg = _cfg
        self._code = None
        
        self._num_embedding_dimensions = int( _cfg['Autoencoders']['OrderbookAutoencoderNumEmbeddingDimensions'])
        self._num_embeddings = int(_cfg['Autoencoders']['OrderbookAutoencoderNumEmbeddings'] )
        self._code_layer_size = int( _cfg['Autoencoders']['OrderbookAutoencoderCodeLayerSize'])
        self._code_scale = float( _cfg['Autoencoders']['OrderbookAutoencoderCodeScale'] )

        self._encoder = TkModel( json.loads(_cfg['Autoencoders']['OrderbookEncoder']) )
        self._decoder = TkModel( json.loads(_cfg['Autoencoders']['OrderbookDecoder']) )

        # re-initialize decoder weights to suppress undesirable peaks
        self._decoder.initWeights( conv_init_mode='xavier_normal', conv_init_gain=0.025 )
        self._vq = VectorQuantizerEMA( num_embeddings=self._num_embeddings, embedding_dim=self._num_embedding_dimensions )

    def code_layer_size(self):
        return self._code_layer_size

    def code(self):
        return self._code
    
    def get_layer_by_parameter(model, target_param):
        for name, param in model.named_parameters():
            # Check if the parameter object matches the target
            if param is target_param:
                # The 'name' is in the format 'layer_name.sub_layer_name.weight'
                # We need to extract the actual module/layer
            
                # Split the name by '.' to get the layer path
                parts = name.split('.')
                # The last part is typically the parameter name itself (e.g., 'weight', 'bias')
                layer_name = '.'.join(parts[:-1])
            
                # Access the module using getattr recursively
                current_module = model
                for part in parts[:-1]:
                    current_module = getattr(current_module, part)
                return current_module
            
        return None

    def get_trainable_parameters(self, conv_weight_decay:float, dense_weight_decay:float):

        encoder_conv_params = []
        encoder_dense_params = []
        encoder_no_decay_params = []
        for name, param in self._encoder.named_parameters():
            if not param.requires_grad:
                continue
            if not any(nd in name for nd in ["bias", "norm"]):
                layer = self.get_layer_by_parameter( param )
                if isinstance( layer, torch.nn.Linear):
                    encoder_dense_params.append(param)
                else:
                    encoder_conv_params.append(param)
            else:
                encoder_no_decay_params.append(param)
        
        decoder_conv_params = []
        decoder_dense_params = []
        decoder_no_decay_params = []
        for name, param in self._decoder.named_parameters():
            if not param.requires_grad:
                continue
            if not any(nd in name for nd in ["bias", "norm"]):
                layer = self.get_layer_by_parameter( param )
                if isinstance( layer, torch.nn.Linear):
                    decoder_dense_params.append(param)
                else:
                    decoder_conv_params.append(param)
            else:
                decoder_no_decay_params.append(param)

        return [
            {"params": encoder_conv_params, "weight_decay": conv_weight_decay},
            {"params": encoder_dense_params, "weight_decay": dense_weight_decay},
            {"params": encoder_no_decay_params, "weight_decay": 0.0},
            {"params": decoder_conv_params, "weight_decay": conv_weight_decay},
            {"params": decoder_dense_params, "weight_decay": dense_weight_decay},
            {"params": decoder_no_decay_params, "weight_decay": 0.0}
        ]

    def encode(self, input):
        z = self._encoder(input)
        _, _, vq_codes = self._vq.quantize(z)
        self._code = vq_codes
        return vq_codes
    
    def decoder_lipschitz_loss(self, z_q, eps=1e-2):
        noise = eps * torch.randn_like(z_q)
        y1 = self._decoder(z_q)
        y2 = self._decoder(z_q + noise)
        return ((y1 - y2) ** 2).mean() / (eps ** 2)
    
    def code_reconstruction_variance_loss(self, y, vq_codes):
        B, C, L = y.shape
        _, K = vq_codes.shape

        loss = 0.0
        count = 0

        for k in range(K):
            same = vq_codes[:, k].unsqueeze(0) == vq_codes[:, k].unsqueeze(1)
            same = same.float()

            diff = y.unsqueeze(0) - y.unsqueeze(1)
            dist = diff.pow(2).mean(dim=(2, 3))

            loss += (same * dist).sum() / (same.sum() + 1e-6)
            count += 1

        return loss / count

    def forward(self, input):
        
        z = self._encoder(input)

        z_q, vq_loss, vq_codes = self._vq(z)
        self._code = vq_codes

        y = self._decoder(z_q)

        # normalize
        y = y - y.min(dim=2, keepdim=True)[0]      # shift ≥ 0
        y = y / (y.max(dim=2, keepdim=True)[0] + 1e-6)  # scale to 0..1

        # decoder loss
        smoothness_loss = self.decoder_lipschitz_loss( z_q, 1.0 ) + self.code_reconstruction_variance_loss( y, vq_codes )
        
        return y, vq_loss, smoothness_loss
