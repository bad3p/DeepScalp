import configparser
import torch
import json
from TkModules.TkModel import TkModel
from torch.distributions import Gamma, Dirichlet

#------------------------------------------------------------------------------------------------------------------------

class TkLastTradesAutoencoder(torch.nn.Module):

    def __init__(self, _cfg: configparser.ConfigParser):

        super().__init__()

        self._cfg = _cfg
        self._code = None

        self._encoder = TkModel(json.loads(_cfg['Autoencoders']['LastTradesEncoder']))
        self._decoder = TkModel(json.loads(_cfg['Autoencoders']['LastTradesDecoder']))

        self._hidden_layer_size = int(_cfg['Autoencoders']['LastTradesAutoencoderHiddenLayerSize'])
        self._code_layer_size = int(_cfg['Autoencoders']['LastTradesAutoencoderCodeLayerSize'])

        self._smoothness_loss_noise_std = 0.01  # TODO: configure

        # Dirichlet parameter layer (log-alpha)
        self._log_alpha_layer = torch.nn.Linear(
            self._hidden_layer_size,
            self._code_layer_size
        )

        self._reparametrization_layer = torch.nn.Linear(
            self._code_layer_size,
            self._hidden_layer_size
        )

        # Dirichlet prior (symmetric)
        self._alpha_prior = float(
            _cfg['Autoencoders'].get('DirichletAlphaPrior', 1.0)
        )

    # --------------------------------------------------------------------------------

    def code_layer_size(self):
        return self._code_layer_size

    def code(self):
        return self._code

    # --------------------------------------------------------------------------------

    def encode(self, input):
        y = self._encoder(input)
        log_alpha = self._log_alpha_layer(y)
        alpha = torch.exp(log_alpha)
        return alpha

    def decode(self, z):
        # Dirichlet reparameterization
        z = self._sample_dirichlet(z)
        z = self._reparametrization_layer(z)
        z = self._decoder(z)
        return z

    # --------------------------------------------------------------------------------

    def _sample_dirichlet(self, alpha):
        gamma_dist = Gamma(alpha, torch.ones_like(alpha))
        g = gamma_dist.rsample()
        z = g / g.sum(dim=1, keepdim=True)
        return z

    # --------------------------------------------------------------------------------

    def forward(self, input):

        y = self._encoder(input)

        log_alpha = self._log_alpha_layer(y).clamp(-10.0, 10.0)
        alpha = torch.exp(log_alpha)

        self._code = alpha

        # Dirichlet reparameterization
        z = self._sample_dirichlet(alpha)

        z = self._reparametrization_layer(z)
        z = self._decoder(z)
        z = z.clamp(0.0, 1.0)

        # Smoothness loss
        eps = self._smoothness_loss_noise_std * torch.randn_like(input)
        y_noise = self._encoder(input + eps)
        alpha_noise = torch.exp(self._log_alpha_layer(y_noise))

        smoothness_loss = torch.mean(
            (alpha - alpha_noise).pow(2)
        )

        return z, alpha, smoothness_loss

    # --------------------------------------------------------------------------------

    @staticmethod
    def kl_divergence_loss(p,q, eps=1e-8):
        m = 0.5 * (p + q)
        return 0.5 * ( 
            torch.nn.functional.kl_div((p+eps).log(), m, reduction="batchmean") +
            torch.nn.functional.kl_div((q+eps).log(), m, reduction="batchmean")
        )

    # --------------------------------------------------------------------------------

    def kl_divergence(self, alpha):
        """
        KL( Dir(alpha) || Dir(alpha_prior) )
        """
        prior = Dirichlet(
            torch.full_like(alpha, self._alpha_prior)
        )
        posterior = Dirichlet(alpha)
        return torch.distributions.kl_divergence(posterior, prior).mean()

    # --------------------------------------------------------------------------------

    def freeze_parameters(self):
        for p in self.parameters():
            p.requires_grad = False

    # --------------------------------------------------------------------------------            

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