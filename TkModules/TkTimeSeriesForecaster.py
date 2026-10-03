import math
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
# Gated residual fusion with Continuous-Time ALiBi
# --------------------------------------------------------------------------------------------------------------

import torch

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
        self.num_heads = num_heads

        # 1) Project heterogeneous inputs
        self.projections = torch.nn.ModuleList([
            torch.nn.Linear(d, embed_dim) for d in input_dims
        ])

        # Context attention
        self.context_attn = torch.nn.Linear(embed_dim, 1)

        # Pre-compute Continuous-Time ALiBi slopes
        # Slopes follow a geometric sequence: m = 2^(-8/num_heads * i)
        slopes = torch.tensor(
            [2 ** (-4 * i / num_heads) for i in range(1, num_heads + 1)], # 8 -> 4
            dtype=torch.float32
        )
        self.register_buffer("alibi_slopes", slopes)

        # Multi-head self-attention across sources
        self.mha = torch.nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Gated residual fusion
        self.gate = torch.nn.Sequential(
            torch.nn.Linear(embed_dim * 2, embed_dim),
            torch.nn.Sigmoid()
        )

        # Pooling
        if pooling == "attn":
            self.pool_attn = torch.nn.Linear(embed_dim, 1)

        self.norm = torch.nn.LayerNorm(embed_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, inputs, pos_encoding=None, log_time_delta=None):

        # Project to shared embedding (Pure Data)
        projected = []
        for proj, h in zip(self.projections, inputs):
            p = proj(h)
            if p.dim() == 2:
                p = p.unsqueeze(1)
            projected.append(p)
            
        # Concatenate along the sequence dimension: (B, N*T, D)
        x = torch.cat(projected, dim=1) 

        # Create position-infused keys/queries
        x_attn = x
        if pos_encoding is not None:
            x_attn = x_attn + pos_encoding.expand(x.size(0), -1, -1)

        # Build global context using pure data
        context_scores = self.context_attn(x).squeeze(-1)
        context_weights = torch.softmax(context_scores, dim=1)
        context = torch.sum(x * context_weights.unsqueeze(-1), dim=1, keepdim=True)
        
        if pos_encoding is not None:
            context_attn = context + pos_encoding.mean(dim=1, keepdim=True) 
        else:
            context_attn = context

        # Continuous-Time ALiBi Attention Mask
        attn_mask = None
        if log_time_delta is not None:
            B = x.size(0)
            N = self.num_sources
            
            # Convert log time delta back to continuous temporal distance
            time_distance = torch.exp(log_time_delta) # (B, T, 1)
            
            # Expand to match concatenated N sources: (B, N*T)
            time_distance = time_distance.repeat(1, N, 1).squeeze(-1)
            
            # Apply geometric slopes to the temporal distance
            # Shape: (B, num_heads, N*T)
            alibi_bias = -self.alibi_slopes.view(1, -1, 1) * time_distance.unsqueeze(1)
            
            # Reshape to expected MultiheadAttention mask shape: (B * num_heads, L, S)
            # L = 1 (context query), S = N*T (keys)
            attn_mask = alibi_bias.view(B * self.num_heads, 1, x.size(1))

        # Contextual attention
        # Q = Context, K = x_attn, V = x
        attn_out, attn_weights = self.mha(
            context_attn, 
            x_attn, 
            x, 
            need_weights=True,
            attn_mask=attn_mask
        )

        # Broadcast attended context back to tokens
        attn_out = attn_out.expand(-1, x.size(1), -1)

        # Gated residual fusion
        gate_input = torch.cat([x, attn_out], dim=-1)
        g = self.gate(gate_input)                     

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
# Continuous EMA Norm module with trainable half-life
# --------------------------------------------------------------------------------------------------------------

class ParallelContinuousEMANorm(torch.nn.Module):
    def __init__(self, num_features, eps=1e-5, init_time_constants=None):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        
        if init_time_constants is not None:
            tau = torch.tensor(init_time_constants, dtype=torch.float32)
            target_lambdas = 1.0 / tau
            if target_lambdas.dim() == 0:
                target_lambdas = target_lambdas.expand(num_features)
        else:
            target_lambdas = torch.full((num_features,), math.log(2.0))
            
        w_init = torch.log(torch.exp(target_lambdas) - 1.0)
        self.w = torch.nn.Parameter(w_init)
        
        self.gamma = torch.nn.Parameter(torch.ones(num_features))
        self.beta = torch.nn.Parameter(torch.zeros(num_features))

    def _parallel_scan(self, u, lam_T):
        """
        Computes the parallel prefix sum in log-space for strictly non-negative inputs.
        """
        # Clamp strictly above 0 to prevent NaN gradients in backward pass
        # The backward pass of clamp_min safely assigns a 0 gradient for inputs < 1e-12
        u_clamped = u.clamp_min(1e-12)
        
        # Directly compute log. We no longer inject -inf for 0-values.
        log_u = torch.log(u_clamped)
        
        # Parallel cumulative sum in log space: log( sum( exp(lam_T + log_u) ) )
        scan = torch.logcumsumexp(lam_T + log_u, dim=1)
        
        # Multiply by exp(-lambda * T) by subtracting in log space, then exponentiate
        return torch.exp(scan - lam_T)

    def forward(self, x, dt, init_mu=None, init_var=None):
        batch_size, seq_len, _ = x.shape
        
        # Reshape lambdas for broadcasting: (1, 1, num_features)
        lambdas = torch.nn.functional.softplus(self.w).view(1, 1, -1)
        
        # Cumulative time vector T_t
        T = torch.cumsum(dt, dim=1)
        lam_T = lambdas * T
        
        # Decay factor for current step: alpha_t
        alpha = torch.exp(-lambdas * dt)
        
        # --- 1. PARALLEL MEAN SCAN ---
        u = (1 - alpha) * x
        
        # We must split 'u' into positive and negative streams because log(-x) is undefined
        mu_pos = self._parallel_scan(torch.relu(u), lam_T)
        mu_neg = self._parallel_scan(torch.relu(-u), lam_T)
        mu = mu_pos - mu_neg
        
        # Apply the initial mean state decay
        if init_mu is None:
            init_mu = x[:, 0, :]
        mu = mu + init_mu.unsqueeze(1) * torch.exp(-lam_T)
        
        # --- 2. PARALLEL VARIANCE SCAN ---
        # With mu computed for all 't', variance v_t is just another non-negative scan
        v = (1 - alpha) * (x - mu)**2
        var = self._parallel_scan(v, lam_T)
        
        if init_var is None:
            init_var = torch.zeros_like(x[:, 0, :])
        var = var + init_var.unsqueeze(1) * torch.exp(-lam_T)
        
        # --- 3. NORMALIZATION ---
        x_norm = (x - mu) / torch.sqrt(var + self.eps)
        x_norm = x_norm * self.gamma + self.beta
        
        # Extract the final hidden states for the next sequence chunk
        final_mu = mu[:, -1, :]
        final_var = var[:, -1, :]
        
        return x_norm, (final_mu, final_var)

# --------------------------------------------------------------------------------------------------------------
# Time series forecasting model
# --------------------------------------------------------------------------------------------------------------

class TkTimeSeriesForecaster(torch.nn.Module):

    def __init__(self, _cfg : configparser.ConfigParser):

        super(TkTimeSeriesForecaster, self).__init__()

        self._cfg = _cfg
        self._num_market_regimes = len(json.loads(_cfg['TimeSeries']['VolatilityRegimes'])) + 1
        self._num_trend_regimes = len( json.loads( _cfg['TimeSeries']['TrendRegimes'] ) ) + 1
        self._prior_steps_count = int(_cfg['TimeSeries']['PriorStepsCount']) 
        self._display_slice = int(_cfg['TimeSeries']['DisplaySlice'])  
        self._input_width = int(_cfg['TimeSeries']['InputWidth'])  
        self._target_width = int(_cfg['Autoencoders']['LastTradesWidth']) 
        self._input_slices = json.loads(_cfg['TimeSeries']['InputSlices'])
        self._log_time_delta_feature_index = int(_cfg['TimeSeries']['LogTimeDeltaFeatureIndex'])
        self._embedding_specification = json.loads(_cfg['TimeSeries']['Embedding'])
        self._embedding_dropout = float(_cfg['TimeSeries']['EmbeddingDropout'])
        self._smm_specification = json.loads(_cfg['TimeSeries']['SMM'])
        self._mlp = TkModel( json.loads(_cfg['TimeSeries']['MLP']) )
        self._regime_mlp = TkModel( json.loads(_cfg['TimeSeries']['RegimeMLP']) )
        self._trend_mlp = TkModel( json.loads(_cfg['TimeSeries']['TrendMLP']) )
        self._fusion_embedding_dims = int(_cfg['TimeSeries']['FusionEmbeddingDims']) 
        self._fusion_attention_heads = int(_cfg['TimeSeries']['FusionAttentionHeads']) 
        self._fusion_dropout = float(_cfg['TimeSeries']['FusionDropout'])    
        self._init_ema_half_life = json.loads(_cfg['TimeSeries']['InitEMAHalfLife'])        

        if len(self._input_slices) != len(self._smm_specification):
            raise RuntimeError('InputSlices and SMM config mismatched!')
        
        self._source_pos_embedding = torch.nn.Parameter( torch.randn(1, len(self._input_slices), self._fusion_embedding_dims) * 0.02 )
        
        self._fusion_input_dims = []

        self._ema_norm = []
        self._embedding = []
        self._smm_proj = []
        self._smm = []
        self._smm_norm = []
        self._mlp_input_size = 0
        for i in range(len(self._input_slices)):
            ch0 = self._input_slices[i][0]
            ch1 = self._input_slices[i][1]
            slice_size = ch1 - ch0

            self._ema_norm.append( ParallelContinuousEMANorm( num_features=slice_size, eps=1e-5, init_time_constants=self._init_ema_half_life[i]) )

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

        self._ema_norm = torch.nn.ModuleList( self._ema_norm )
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
            self._fusion.gate[0].bias.fill_(0.0) # -2.0

        # reinitialize MLP weights
        for m in self._mlp.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                torch.nn.init.constant_(m.bias, 0)        
        
        self._smm_output_tensors = None

    def get_trainable_parameters(self, embedding_weight_decay:float, smm_weight_decay:float, fusion_weight_decay:float, mlp_weight_decay:float, embedding_learning_rate:float, smm_learning_rate:float, smm_ev_learning_rate:float, smm_dt_learning_rate:float, fusion_learning_rate:float, mlp_learning_rate:float, ema_learning_rate:float):

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

        for name, param in self._mlp.named_parameters():
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

        for name, param in self._trend_mlp.named_parameters():
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

        other = []
        for ema_norm in self._ema_norm:
            other.append( ema_norm.w )
            other.append( ema_norm.gamma )
            other.append( ema_norm.beta )

        return [
            {"params": embedding_decay_params, "weight_decay": embedding_weight_decay, 'lr': embedding_learning_rate}, #0
            {"params": embedding_no_decay_params, "weight_decay": 0.0, 'lr': embedding_learning_rate}, #1
            {"params": smm_decay_params, "weight_decay": smm_weight_decay, 'lr': smm_learning_rate}, #2
            {"params": smm_ev_params, "weight_decay": 0.0, 'lr': smm_ev_learning_rate}, #3
            {"params": smm_dt_params, "weight_decay": 0.0, 'lr': smm_dt_learning_rate}, #4
            {"params": smm_no_decay_params, "weight_decay": 0.0, 'lr': smm_learning_rate}, #5
            {"params": mlp_decay_params, "weight_decay": mlp_weight_decay, 'lr': mlp_learning_rate}, #6
            {"params": mlp_no_decay_params, "weight_decay": 0.0, 'lr': mlp_learning_rate}, #7
            {"params": fusion_decay_params, "weight_decay": fusion_weight_decay, 'lr': fusion_learning_rate}, #8
            {"params": fusion_no_decay_params, "weight_decay": 0.0, 'lr': fusion_learning_rate}, #9
            {"params": other, "weight_decay": 0.0, 'lr': ema_learning_rate}, #10
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

    def ema_group_indices(self):
        return [10]

    def ema_half_life(self):

        result = []
        for ema_norm in self._ema_norm:
            w = ema_norm.w.detach()
            w = torch.exp( w )
            w = w + 1
            w = torch.log( w)
            w = 1.0 / w
            
            result.extend( list(w) )
        return result

    def smm_output(self):
        return self._smm_output_tensors

    def input_slice(self, idx:int):
        return self._input_slice_tensors[idx]

    def forward(self, input):

        batch_size = input.shape[0]

        input = torch.reshape( input, ( batch_size, self._prior_steps_count, self._input_width) )

        log_time_delta = input[:, :, self._log_time_delta_feature_index : self._log_time_delta_feature_index + 1]
        time_delta = torch.exp( log_time_delta )
        log_time_delta = log_time_delta - 4.605 # TODO: configure; 4.605 = ln(100) = ln(max_observed_time)

        self._input_slice_tensors = []
        self._smm_output_tensors = []

        for i in range(len(self._input_slices)):
            ch0 = self._input_slices[i][0]
            ch1 = self._input_slices[i][1]
            slice_size = ch1 - ch0
            input_slice_tensor = input[:, :, ch0:ch1]

            input_slice_tensor, _ = self._ema_norm[i]( input_slice_tensor, time_delta )
            
            self._input_slice_tensors.append( input_slice_tensor )

            input_slice_tensor = self._embedding[i]( input_slice_tensor )
            
            x = self._smm_proj[i]( input_slice_tensor )
            for ssm, norm in zip(self._smm[i], self._smm_norm[i]):
                residual = x
                x = norm(x)
                x = ssm(x, log_time_delta)
                x = torch.nn.functional.silu(x)
                x = x + residual            
            x = self._smm_norm[i][-1]( x )
            self._smm_output_tensors.append( x )

        # Add dummy temporal dimension: (1, N, D) -> (1, N, 1, D)
        # Expand it identically across all T time steps: (1, N, T, D)
        T = self._prior_steps_count
        N = len(self._input_slices)
        pos_emb = self._source_pos_embedding.unsqueeze(2).expand(1, N, T, self._fusion_embedding_dims)

        # Flatten N and T together to perfectly match the fusion layer's (B, N*T, D) shape
        pos_emb = pos_emb.reshape(1, N * T, self._fusion_embedding_dims)

        # no fusion case
        # merged = torch.cat( self._smm_output_tensors, dim=-1)
        fused, source_attn, gates = self._fusion(self._smm_output_tensors, pos_emb, log_time_delta )
        merged = fused

        # monitoring feedback
        self._smm_output_tensors = fused

        # regime prediction branch
        y_regime = self._regime_mlp.forward( merged )
        y_regime = torch.reshape( y_regime, (y_regime.shape[0], self._num_market_regimes ) )

        # trend prediction branch
        y_trend_regime = self._trend_mlp.forward( merged )
        y_trend_regime = torch.reshape( y_trend_regime, (y_trend_regime.shape[0], self._num_trend_regimes ) )

        y = self._mlp( merged )
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

        return y, y_regime, y_trend_regime

    @staticmethod    
    def kl_divergence_from_logits(logits, target, eps=1e-8):
        log_preds = torch.nn.functional.log_softmax(logits, dim=-1)
        # reduction='none' returns a tensor of shape (batch_size, num_bins)
        kld_matrix = torch.nn.functional.kl_div(log_preds, target, reduction='none')    
        # Sum across the bins to get the total KLD for each individual sample
        kld_loss = torch.sum(kld_matrix, dim=-1) # Shape: (batch_size,)
        return kld_loss

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
    
    @staticmethod
    def emd_mse_1d_from_logits(logits, target):
    
        q = torch.nn.functional.softmax(logits, dim=-1)

        # cumulative distributions
        cdf_q = torch.cumsum(q, dim=-1)
        cdf_t = torch.cumsum(target, dim=-1)

        emd_mse = (cdf_q - cdf_t) ** 2
        emd_mse = emd_mse.sum(dim=-1)
        return emd_mse

    @staticmethod
    def gaussian_smoothing_1d(x: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
        """
        Applies 1D Gaussian smoothing to a tensor of shape [B, W].
    
        Args:
            x (torch.Tensor): Input one-hot or dense tensor of shape [B, W].
            kernel_size (int): The total width of the Gaussian window (should be odd).
            sigma (float): Standard deviation of the Gaussian distribution.
        
        Returns:
            torch.Tensor: Smoothed tensor of shape [B, W].
        """
        # 1. Create a 1D grid centered at 0
        radius = kernel_size // 2
        grid = torch.arange(-radius, radius + 1, dtype=torch.float32, device=x.device)
    
        # 2. Compute the 1D Gaussian kernel
        # formula: exp(-x^2 / (2 * sigma^2))
        kernel = torch.exp(-grid**2 / (2 * sigma**2))
        kernel = kernel / kernel.sum()  # Normalize to sum to 1
    
        # 3. Reshape kernel for conv1d: [out_channels, in_channels, kernel_width]
        # We want 1 input channel and 1 output channel
        kernel = kernel.view(1, 1, -1)
    
        # 4. Reshape input tensor [B, W] -> [B, C, W] where C=1
        x_unsqueezed = x.unsqueeze(1).float()
    
        # 5. Apply padding to maintain the dimension W
        # Same padding for a given kernel size is equal to the radius
        padding_size = radius
    
        # 6. Perform 1D convolution
        smoothed = torch.nn.functional.conv1d(x_unsqueezed, kernel, padding=padding_size)
    
        # 7. Squeeze the channel dimension back to match original shape [B, W]
        smoothed_sq = smoothed.squeeze(1)
        
        # 8. Normalize the output so the sum of the spatial dimension (W) is 1
        # 1e-8 is added to prevent division by zero for empty rows
        normalized = smoothed_sq / (smoothed_sq.sum(dim=-1, keepdim=True) + 1e-8)
        
        return normalized