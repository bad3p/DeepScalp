
import configparser
import torch
import json
from torch.distributions import Gamma, Dirichlet
from TkModules.TkModel import TkModel
from TkModules.TkStackedLSTM import TkStackedLSTM
from TkModules.TkSelfAttention import TkSelfAttention
from TkModules.TkTCNN import TkTCNN
from TkModules.TkStateSpace import TkStateSpaceModule

# --------------------------------------------------------------------------------------------------------------
# Gated residual fusion
# --------------------------------------------------------------------------------------------------------------

class MultiHeadFusionGRF(torch.nn.Module):
    def __init__(
        self,
        input_dims,
        embed_dim=128,
        num_heads=4,
        dropout=0.1,
        pooling="attn"
    ):
        super().__init__()

        assert embed_dim % num_heads == 0

        self.pooling = pooling
        self.num_sources = len(input_dims)

        # 1) Project heterogeneous inputs
        self.projections = torch.nn.ModuleList([
            torch.nn.Linear(d, embed_dim) for d in input_dims
        ])

        # 1.1) Context attention
        self.context_attn = torch.nn.Linear(embed_dim, 1)

        # 2) Multi-head self-attention across sources
        self.mha = torch.nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 3) Gated residual fusion
        self.gate = torch.nn.Sequential(
            torch.nn.Linear(embed_dim * 2, embed_dim),
            torch.nn.Sigmoid()
        )

        # 4) Pooling
        if pooling == "attn":
            self.pool_attn = torch.nn.Linear(embed_dim, 1)

        self.norm = torch.nn.LayerNorm(embed_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, inputs, pos_encoding = None):

        # 1. Project to shared embedding (Pure Data)
        x = torch.stack(
            [proj(h) for proj, h in zip(self.projections, inputs)],
            dim=1
        )  # (B, N, D)

        # 2. Create position-infused keys/queries
        if pos_encoding is not None:
            # Simple addition is mathematically equivalent to your previous linear collapse, 
            # but much more efficient.
            x_attn = x + pos_encoding.expand(x.size(0), -1, -1)
        else:
            x_attn = x

        # 3. Build global context using pure data
        context_scores = self.context_attn(x).squeeze(-1)
        context_weights = torch.softmax(context_scores, dim=1)
        context = torch.sum(x * context_weights.unsqueeze(-1), dim=1, keepdim=True)
        
        # (Optional) Inject position into the context if you want the query to know "where" it came from
        if pos_encoding is not None:
            context_attn = context + pos_encoding.mean(dim=1, keepdim=True) # or a learned global pos
        else:
            context_attn = context

        # 4. Contextual attention
        # Q = Context (with pos), K = x_attn (with pos), V = x (PURE DATA)
        attn_out, attn_weights = self.mha(context_attn, x_attn, x, need_weights=True)

        # Broadcast attended context back to tokens
        attn_out = attn_out.expand(-1, x.size(1), -1)  # (B, N, D)

        # 5. Gated residual fusion (using pure data for the residual)
        gate_input = torch.cat([x, attn_out], dim=-1)  # (B, N, 2D)
        g = self.gate(gate_input)                      # (B, N, D)

        fused_tokens = g * attn_out + (1.0 - g) * x

        # Normalize & Dropout
        fused_tokens = self.norm(fused_tokens)
        fused_tokens = self.dropout(fused_tokens)

        # Pool across inputs
        if self.pooling == "mean":
            fused = fused_tokens.mean(dim=1)

        elif self.pooling == "max":
            fused, _ = fused_tokens.max(dim=1)

        elif self.pooling == "attn":
            scores = self.pool_attn(fused_tokens).squeeze(-1)
            weights = torch.softmax(scores, dim=1)
            fused = torch.sum(
                fused_tokens * weights.unsqueeze(-1),
                dim=1
            )

        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        return fused, attn_weights, g

# --------------------------------------------------------------------------------------------------------------
# Embedding for VQ-VAE codes
# Embeds a N-D VQ-VAE code (each dim in [0, K]) into a continuous vector.
# Input:  (B, T, N)  or (B, N)
# Output: (B, T, D)  or (B, D)
# --------------------------------------------------------------------------------------------------------------

class VQCodeEmbedding(torch.nn.Module):
    def __init__(
        self,
        num_codes: int = 256,
        code_dim: int = 16,
        embed_dim: int = 32,
        hidden_dim: int = 256,
        out_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.code_dim = code_dim

        # Shared embedding table 
        self.embedding = torch.nn.Embedding(num_codes, embed_dim)

        # MLP fusion
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(code_dim * embed_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, out_dim),
            torch.nn.LayerNorm(out_dim),
        )

        self._init_weights()

    def _init_weights(self):
        torch.nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        for m in self.mlp:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        
        # codes: LongTensor of shape (B, T, N) or (B, N)        
        orig_shape = codes.shape

        if codes.dim() == 2:
            # (B, N) -> (B, 1, N)
            codes = codes.unsqueeze(1)

        B, T, D = codes.shape
        assert D == self.code_dim, f"Expected {self.code_dim} code dims, got {D}"

        # Embed each code index
        # (B, T, N) -> (B, T, N, embed_dim)
        x = self.embedding(codes.long())

        # Flatten code positions
        # (B, T, N * embed_dim)
        x = x.view(B, T, -1)

        # Fuse
        # (B, T, out_dim)
        x = self.mlp(x)

        if len(orig_shape) == 2:
            # Return (B, out_dim)
            x = x.squeeze(1)

        return x

# --------------------------------------------------------------------------------------------------------------
# Embedding for scalar group
# --------------------------------------------------------------------------------------------------------------

class ScalarGroupEmbedding(torch.nn.Module):
    def __init__(self, in_channels, specification:list, dropout:float):
        super().__init__()
        self._proj = []
        for i in range(len(specification)):
            self._proj.append( torch.nn.Linear( in_channels if i == 0 else specification[i-1], specification[i] ) )
            self._proj.append( torch.nn.SiLU() )
        self._proj.append( torch.nn.Dropout(dropout) )
        self._proj = torch.nn.ModuleList( self._proj )        
        self._init_weights()

    def _init_weights(self):        
        for m in self._proj:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        for _,layer in enumerate(self._proj):
            x = layer(x)
        return x

# --------------------------------------------------------------------------------------------------------------
# Time series forecasting model
# --------------------------------------------------------------------------------------------------------------

class TkTimeSeriesForecaster(torch.nn.Module):

    def __init__(self, _cfg : configparser.ConfigParser):

        super(TkTimeSeriesForecaster, self).__init__()

        self._cfg = _cfg
        self._num_market_regimes = int(_cfg['TimeSeries']['NumMarketRegimes'])
        self._prior_steps_count = int(_cfg['TimeSeries']['PriorStepsCount']) 
        self._display_slice = int(_cfg['TimeSeries']['DisplaySlice'])  
        self._input_width = int(_cfg['TimeSeries']['InputWidth'])  
        self._target_width = int(_cfg['Autoencoders']['LastTradesWidth']) 
        self._input_slices = json.loads(_cfg['TimeSeries']['InputSlices'])
        self._embedding_specification = json.loads(_cfg['TimeSeries']['Embedding'])
        self._embedding_dropout = float(_cfg['TimeSeries']['EmbeddingDropout'])
        self._smm_specification = json.loads(_cfg['TimeSeries']['SMM'])
        self._encoder_mlp = TkModel( json.loads(_cfg['TimeSeries']['EncoderMLP']) )
        self._decoder_mlp = TkModel( json.loads(_cfg['TimeSeries']['DecoderMLP']) )
        self._autoencoder_hidden_layer_size = int(_cfg['TimeSeries']['AutoencoderHiddenLayerSize'])
        self._autoencoder_code_layer_size = int(_cfg['TimeSeries']['AutoencoderCodeLayerSize'])
        self._autoencoder_free_bits_threshold = float(_cfg['TimeSeries']['AutoencoderFreeBitsThreshold'])
        self._regime_mlp = TkModel( json.loads(_cfg['TimeSeries']['RegimeMLP']) )        
        self._fusion_embedding_dims = int(_cfg['TimeSeries']['FusionEmbeddingDims']) 
        self._fusion_attention_heads = int(_cfg['TimeSeries']['FusionAttentionHeads']) 
        self._fusion_dropout = float(_cfg['TimeSeries']['FusionDropout'])    

        # VAE layers
        self._autoencoder_mu_layer = torch.nn.Linear(self._autoencoder_hidden_layer_size, self._autoencoder_code_layer_size)
        self._autoencoder_logvar_layer = torch.nn.Linear(self._autoencoder_hidden_layer_size, self._autoencoder_code_layer_size)
        self._autoencoder_reparametrization_layer = torch.nn.Linear( self._autoencoder_code_layer_size + self._num_market_regimes, self._autoencoder_hidden_layer_size )
        #self._autoencoder_reparametrization_layer = torch.nn.Linear( self._autoencoder_code_layer_size, self._autoencoder_hidden_layer_size )

        if len(self._input_slices) != len(self._smm_specification):
            raise RuntimeError('InputSlices and SMM config mismatched!')
        self._source_pos_embedding = torch.nn.Parameter( torch.randn(1, len(self._input_slices), self._fusion_embedding_dims) * 0.02 )
        
        self._fusion_input_dims = []

        self._embedding = []
        self._smm_proj = []
        self._smm = []
        self._smm_norm = []
        self._mlp_input_size = 0
        for i in range(len(self._input_slices)):
            ch0 = self._input_slices[i][0]
            ch1 = self._input_slices[i][1]
            slice_size = ch1 - ch0

            if not self._embedding_specification[i]:
                self._embedding.append( torch.nn.Identity() )
            else:
                embedding_specification = self._embedding_specification[i]
                embedding_type = list(embedding_specification.keys())[0]
                embedding_descriptor = embedding_specification[embedding_type]
                if embedding_type == 'Lookup':
                    num_codes = embedding_descriptor[0]
                    code_dim = embedding_descriptor[1]
                    embed_dim = embedding_descriptor[2]
                    hidden_dim = embedding_descriptor[3]
                    out_dim = embedding_descriptor[4]
                    dropout = embedding_descriptor[5]
                    self._embedding.append( VQCodeEmbedding( num_codes, code_dim, embed_dim, hidden_dim, out_dim, dropout ) )
                    slice_size = out_dim
                elif embedding_type == 'MLP':
                    self._embedding.append( ScalarGroupEmbedding( slice_size, embedding_descriptor, self._embedding_dropout ) )
                    slice_size = embedding_descriptor[-1]
                else:
                    raise RuntimeError('Unknown embedding type:'+embedding_type)

            state_size = self._smm_specification[i][0]
            model_size = self._smm_specification[i][1]
            num_layers = self._smm_specification[i][2]
            self._smm_proj.append( torch.nn.Linear( slice_size, model_size) )
            self._smm.append( torch.nn.ModuleList( [ TkStateSpaceModule( model_size, state_size, model_size) for _ in range(num_layers) ] ) )
            self._smm_norm.append( torch.nn.ModuleList( [ torch.nn.LayerNorm( model_size ) for _ in range(num_layers+1) ] ) )
            self._mlp_input_size = self._mlp_input_size + model_size
            self._fusion_input_dims.append( model_size )

        self._embedding = torch.nn.ModuleList( self._embedding )
        self._smm_proj = torch.nn.ModuleList( self._smm_proj )
        self._smm = torch.nn.ModuleList( self._smm )
        self._smm_norm = torch.nn.ModuleList( self._smm_norm )

        self._fusion = MultiHeadFusionGRF( 
            input_dims=self._fusion_input_dims, 
            embed_dim=self._fusion_embedding_dims, 
            num_heads=self._fusion_attention_heads, 
            dropout=self._fusion_dropout, 
            pooling="attn"
        )

        # enforce gate bias
        with torch.no_grad():
            self._fusion.gate[0].bias.fill_(-2.0)

        # reinitialize MLP weights
        for m in self._encoder_mlp.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                torch.nn.init.constant_(m.bias, 0)
        for m in self._decoder_mlp.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                torch.nn.init.constant_(m.bias, 0)

        # learnable normalization temperature
        # self._y_scale = torch.nn.Parameter(torch.tensor(5.0))
        # self._y_regime_scale = torch.nn.Parameter(torch.tensor(5.0))
        
        self._smm_output_tensors = None

    def get_trainable_parameters(self, embedding_weight_decay:float, smm_weight_decay:float, fusion_weight_decay:float, mlp_weight_decay:float, embedding_learning_rate:float, smm_learning_rate:float, smm_ev_learning_rate:float, smm_dt_learning_rate:float, fusion_learning_rate:float, mlp_learning_rate:float):

        embedding_decay_params = []
        embedding_no_decay_params = []

        smm_no_decay_params = list(self._smm_norm.parameters())
        smm_decay_params = list(self._smm_proj.parameters())
        smm_ev_params = []
        smm_dt_params = []

        for smm_module in self._smm:
            for smm in smm_module:
                smm_ev_params.append( smm.log_lambda_real )
                smm_ev_params.append( smm.lambda_imag )
                smm_dt_params.append( smm.log_dt )
                smm_decay_params.append( smm.B )
                smm_decay_params.append( smm.C_real )
                smm_decay_params.append( smm.C_imag )
        
        mlp_decay_params = []
        mlp_no_decay_params = []

        fusion_decay_params = [ ]
        fusion_no_decay_params = [ self._source_pos_embedding ]

        for name, param in self._embedding.named_parameters():
            if not param.requires_grad:
                continue
            if not any(nd in name for nd in ["bias", "norm"]):
                embedding_decay_params.append(param)
            else:
                embedding_no_decay_params.append(param)

        for name, param in self._encoder_mlp.named_parameters():
            if not param.requires_grad:
                continue
            if not any(nd in name for nd in ["bias", "norm"]):
                mlp_decay_params.append(param)
            else:
                mlp_no_decay_params.append(param)

        for name, param in self._decoder_mlp.named_parameters():
            if not param.requires_grad:
                continue
            if not any(nd in name for nd in ["bias", "norm"]):
                mlp_decay_params.append(param)
            else:
                mlp_no_decay_params.append(param)

        for name, param in self._autoencoder_mu_layer.named_parameters():
            if not param.requires_grad:
                continue
            if not any(nd in name for nd in ["bias", "norm"]):
                mlp_decay_params.append(param)
            else:
                mlp_no_decay_params.append(param)

        for name, param in self._autoencoder_logvar_layer.named_parameters():
            if not param.requires_grad:
                continue
            if not any(nd in name for nd in ["bias", "norm"]):
                mlp_decay_params.append(param)
            else:
                mlp_no_decay_params.append(param)

        for name, param in self._autoencoder_reparametrization_layer.named_parameters():
            if not param.requires_grad:
                continue
            if not any(nd in name for nd in ["bias", "norm"]):
                mlp_decay_params.append(param)
            else:
                mlp_no_decay_params.append(param)

        for name, param in self._regime_mlp.named_parameters():
            if not param.requires_grad:
                continue
            if not any(nd in name for nd in ["bias", "norm"]):
                mlp_decay_params.append(param)
            else:
                mlp_no_decay_params.append(param)

        for name, param in self._fusion.named_parameters():
            if not param.requires_grad:
                continue
            if not any(nd in name for nd in ["bias", "norm"]):
                fusion_decay_params.append(param)
            else:
                fusion_no_decay_params.append(param)

        # other = [ ]

        return [
            {"params": embedding_decay_params, "weight_decay": embedding_weight_decay, 'lr': embedding_learning_rate}, #0
            {"params": embedding_no_decay_params, "weight_decay": 0.0, 'lr': embedding_learning_rate}, #1
            {"params": smm_decay_params, "weight_decay": smm_weight_decay, 'lr': smm_learning_rate}, #2
            {"params": smm_ev_params, "weight_decay": smm_weight_decay, 'lr': smm_ev_learning_rate}, #3
            {"params": smm_dt_params, "weight_decay": smm_weight_decay, 'lr': smm_dt_learning_rate}, #4
            {"params": smm_no_decay_params, "weight_decay": 0.0, 'lr': smm_learning_rate}, #5
            {"params": mlp_decay_params, "weight_decay": mlp_weight_decay, 'lr': mlp_learning_rate}, #6
            {"params": mlp_no_decay_params, "weight_decay": 0.0, 'lr': mlp_learning_rate}, #7
            {"params": fusion_decay_params, "weight_decay": fusion_weight_decay, 'lr': fusion_learning_rate}, #8
            {"params": fusion_no_decay_params, "weight_decay": 0.0, 'lr': fusion_learning_rate}, #9
            # {"params": other, "weight_decay": 0.0, 'lr': mlp_learning_rate},
        ]
    
    def embedding_group_indices(self):
        return [0,1]
    
    def embedding_decay_group_indices(self):
        return [0]
    
    def smm_group_indices(self):
        return [2,5]
    
    def smm_ev_group_indices(self):
        return [3]
    
    def smm_dt_group_indices(self):
        return [4]
    
    def smm_decay_group_indices(self):
        return [2,3,4]
        
    def mlp_group_indices(self):
        return [6,7]
    
    def mlp_decay_group_indices(self):
        return [6]

    def fusion_group_indices(self):
        return [8,9]
    
    def fusion_decay_group_indices(self):
        return [8]

    def smm_output(self):
        return self._smm_output_tensors

    def input_slice(self, idx:int):
        return self._input_slice_tensors[idx]

    def forward(self, input):

        batch_size = input.shape[0]

        input = torch.reshape( input, ( batch_size, self._prior_steps_count, self._input_width) )

        self._input_slice_tensors = []
        self._smm_output_tensors = []

        for i in range(len(self._input_slices)):
            ch0 = self._input_slices[i][0]
            ch1 = self._input_slices[i][1]
            slice_size = ch1 - ch0
            input_slice_tensor = input[:, :, ch0:ch1]
            input_slice_tensor = self._embedding[i]( input_slice_tensor )
            self._input_slice_tensors.append( input_slice_tensor )
            
            x = self._smm_proj[i]( input_slice_tensor )
            for ssm, norm in zip(self._smm[i], self._smm_norm[i]):
                residual = x
                x = norm(x)
                x = ssm(x)
                x = torch.nn.functional.silu(x)
                x = x + residual            
            x = self._smm_norm[i][-1]( x )
            smm_output = x[:, -1, :]
            self._smm_output_tensors.append( smm_output )

        # no fusion case
        # merged = torch.cat( self._smm_output_tensors, dim=-1)
        fused, source_attn, gates = self._fusion(self._smm_output_tensors, self._source_pos_embedding )
        merged = fused

        # regime prediction branch
        y_regime = self._regime_mlp.forward( merged )
        y_regime = torch.reshape( y_regime, (y_regime.shape[0], self._num_market_regimes ) )
        y_regime_probs = torch.nn.functional.softmax(y_regime, dim=-1)

        y_encoded = self._encoder_mlp.forward( merged )

        mu = self._autoencoder_mu_layer(y_encoded)
        logvar = self._autoencoder_logvar_layer(y_encoded)
        y_kld_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        y_kld_loss = torch.clamp(y_kld_per_dim, min=self._autoencoder_free_bits_threshold).sum(dim=-1).mean()

        # VAE reparameterization
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu  # Deterministic expected value for inference

        # Detach prevents the reconstruction loss from ruining the regime classifier.
        # We only want the true labels (via cross-entropy) training the _regime_mlp.
        condition = y_regime_probs.detach()
        z_conditioned = torch.cat([z, condition], dim=-1)

        # monitoring feedback
        self._smm_output_tensors = z_conditioned
        # self._smm_output_tensors = z

        y = self._autoencoder_reparametrization_layer(z_conditioned)
        y = self._decoder_mlp(y)
        y = torch.reshape( y, (y.shape[0],y.shape[1]*y.shape[2]))

        #if not self.training:
        #    y = torch.nn.functional.softmax(y, dim=1)
        #    y_regime = torch.nn.functional.softmax(y_regime, dim=1)

        # monitoring feedback
        # if self.training:
        #     self._smm_output_tensors = merged
        # else:
        #     self._smm_output_tensors = torch.cat( self._smm_output_tensors, dim=-1)
        # self._smm_output_tensors = gates
        # self._smm_output_tensors = self._smm_output_tensors[self._display_slice]

        return y, y_regime, y_kld_loss

    @staticmethod    
    def js_divergence_from_logits(logits, target, eps=1e-8):

        # Ensure numerical safety for target
        target = target.clamp(min=eps)
        target = target / target.sum(dim=-1, keepdim=True)

        # Predicted distribution
        q = torch.nn.functional.softmax(logits, dim=-1)

        # Mixture distribution
        m = 0.5 * (target + q)
        m = m.clamp(min=eps)

        # KL terms
        kl_pm = (target * (torch.log(target) - torch.log(m))).sum(dim=-1)
        kl_qm = (q * (torch.log(q + eps) - torch.log(m))).sum(dim=-1)

        js = 0.5 * (kl_pm + kl_qm)

        return js
    
    @staticmethod
    def emd_1d_from_logits(logits, target):
    
        q = torch.nn.functional.softmax(logits, dim=-1)

        # cumulative distributions
        cdf_q = torch.cumsum(q, dim=-1)
        cdf_t = torch.cumsum(target, dim=-1)

        emd = torch.abs(cdf_q - cdf_t).sum(dim=-1)
        return emd

