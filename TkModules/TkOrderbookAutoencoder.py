import configparser
import torch
import json
import math
from TkModules.TkModel import TkModel
from TkModules.TkVectorQuantizerEMA import TkVectorQuantizerEMA

#------------------------------------------------------------------------------------------------------------------------

class TkOrderbookAutoencoder(torch.nn.Module):

    def __init__(self, _cfg : configparser.ConfigParser):

        super(TkOrderbookAutoencoder, self).__init__()

        self._cfg = _cfg
        self._code = None
        
        self._orderbook_regularization_channel_0 = int(_cfg['Autoencoders']['OrderbookRegularizationChannel0'])
        self._orderbook_regularization_channel_1 = int(_cfg['Autoencoders']['OrderbookRegularizationChannel1'])
        self._orderbook_log_volume_branch_injection_layer = int(_cfg['Autoencoders']['OrderbookLogVolumeBranchInjectionLayer'])
        self._orderbook_log_volume_branch_features = int(_cfg['Autoencoders']['OrderbookLogVolumeBranchFeatures'])

        self._num_embedding_dimensions = int( _cfg['Autoencoders']['OrderbookAutoencoderNumEmbeddingDimensions'])
        self._num_embeddings = int(_cfg['Autoencoders']['OrderbookAutoencoderNumEmbeddings'] )
        self._code_layer_size = int( _cfg['Autoencoders']['OrderbookAutoencoderCodeLayerSize'])
        self._code_scale = float( _cfg['Autoencoders']['OrderbookAutoencoderCodeScale'] )

        self._encoder = TkModel( json.loads(_cfg['Autoencoders']['OrderbookEncoder']) )
        self._decoder = TkModel( json.loads(_cfg['Autoencoders']['OrderbookDecoder']) )

        # re-initialize decoder weights to suppress undesirable peaks
        #self._decoder.initWeights( conv_init_mode='xavier_normal', conv_init_gain=0.1 )
        self._vq = TkVectorQuantizerEMA( num_embeddings=self._num_embeddings, embedding_dim=self._num_embedding_dimensions )

        # LOB volume regularization branch
        self._log_volume_head = torch.nn.Linear( self._orderbook_log_volume_branch_features, 1 )

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
    
    def kl_code_usage_loss(self, vq_codes, eps=1e-8):

        if vq_codes.dim() == 2:
            codes = vq_codes.reshape(-1)
        else:
            codes = vq_codes

        # Count code usage
        counts = torch.bincount(codes, minlength=self._num_embeddings).float()

        # Empirical distribution
        p = counts / (counts.sum() + eps)

        # Uniform prior
        p_prior = torch.full_like(p, 1.0 / self._num_embeddings)

        # KL(p || prior)
        kl = torch.sum(p * torch.log((p + eps) / (p_prior + eps)))

        return kl
        
    def forward(self, input):
        
        z = self._encoder(input)
        z = z / (z.norm(dim=1, keepdim=True) + 1e-6)

        z_q, vq_loss, vq_codes = self._vq(z)
        self._code = vq_codes

        y = self._decoder(z_q)
        y = torch.sigmoid(y)

        # LOB volume regularization loss
        y_log_volume = self._decoder.layer_outputs()[ self._orderbook_log_volume_branch_injection_layer ]
        y_log_volume = torch.mean( y_log_volume, dim=2 ) 
        y_log_volume = self._log_volume_head( y_log_volume )
        
        y_log_volume_target_0 = input[:, (self._orderbook_regularization_channel_0):(self._orderbook_regularization_channel_0+1), :]
        y_log_volume_target_0 = torch.sum( y_log_volume_target_0, dim=-1, keepdim=True)
        y_log_volume_target_0 = y_log_volume_target_0 + 1e-7

        y_log_volume_target_1 = input[:, (self._orderbook_regularization_channel_1):(self._orderbook_regularization_channel_1+1), :]
        y_log_volume_target_1 = torch.sum( y_log_volume_target_1, dim=-1, keepdim=True)
        y_log_volume_target_1 = y_log_volume_target_1 + 1e-7

        y_log_volume_target = y_log_volume_target_0 + y_log_volume_target_1

        y_log_volume_target = torch.log( y_log_volume_target )
        y_log_volume_loss = y_log_volume_target - y_log_volume
        y_log_volume_loss = y_log_volume_loss ** 2

        y_kl_loss = self.kl_code_usage_loss( vq_codes )
        
        return y, y_log_volume_loss, vq_loss, y_kl_loss

    @torch.no_grad()
    def get_code_usage(self, eps=1e-8):

        codes = self._code.reshape(-1)

        counts = torch.bincount(
            codes,
            minlength=self._num_embeddings
        ).float()

        total = counts.sum()

        probs = counts / (total + eps)

        # Stats
        active_codes = (counts > 0).sum().item()
        dead_codes = self._num_embeddings - active_codes

        entropy = -(probs * torch.log(probs + eps)).sum().item()
        max_entropy = math.log(self._num_embeddings)
        entropy_norm = entropy / max_entropy

        return active_codes, dead_codes, entropy_norm