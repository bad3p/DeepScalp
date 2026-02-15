
import configparser
import torch
import json
from TkModules.TkModel import TkModel
from TkModules.TkStackedLSTM import TkStackedLSTM
from TkModules.TkSelfAttention import TkSelfAttention
from TkModules.TkTCNN import TkTCNN

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

    def forward(self, inputs):

        # Project to shared embedding
        x = torch.stack(
            [proj(h) for proj, h in zip(self.projections, inputs)],
            dim=1
        )  # (B, N, D)

        # Build global context 
        context_scores = self.context_attn(x).squeeze(-1)
        context_weights = torch.softmax(context_scores, dim=1)
        context = torch.sum(x * context_weights.unsqueeze(-1), dim=1, keepdim=True)

        # Contextual attention
        # Q = context, K/V = sources
        attn_out, attn_weights = self.mha(context, x, x, need_weights=True)

        # Broadcast attended context back to tokens
        attn_out = attn_out.expand(-1, x.size(1), -1)  # (B, N, D)

        # Gated residual fusion
        gate_input = torch.cat([x, attn_out], dim=-1)  # (B, N, 2D)
        g = self.gate(gate_input)                      # (B, N, D)

        fused_tokens = g * attn_out + (1.0 - g) * x

        # Normalize
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
            self._proj.append( torch.nn.LayerNorm(specification[i]) )
            if i < len(specification) - 1:
                self._proj.append( torch.nn.GELU() )
            if i < len(specification) - 1:
                self._proj.append( torch.nn.Dropout(dropout) )
        self._proj = torch.nn.ModuleList( self._proj )        

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
        self._prior_steps_count = int(_cfg['TimeSeries']['PriorStepsCount']) 
        self._display_slice = int(_cfg['TimeSeries']['DisplaySlice'])  
        self._input_width = int(_cfg['TimeSeries']['InputWidth'])  
        self._target_width = int(_cfg['Autoencoders']['LastTradesWidth']) 
        self._input_slices = json.loads(_cfg['TimeSeries']['InputSlices'])
        self._embedding_specification = json.loads(_cfg['TimeSeries']['Embedding'])
        self._lstm_specification = json.loads(_cfg['TimeSeries']['LSTM'])
        self._mlp = TkModel( json.loads(_cfg['TimeSeries']['MLP']) )
        self._aux_loss_slice = int(_cfg['TimeSeries']['AuxLossSlice']) 
        self._aux_mlp = TkModel( json.loads(_cfg['TimeSeries']['AuxMLP']) )
        self._fusion_embedding_dims = int(_cfg['TimeSeries']['FusionEmbeddingDims']) 
        self._fusion_attention_heads = int(_cfg['TimeSeries']['FusionAttentionHeads']) 
        self._fusion_dropout = float(_cfg['TimeSeries']['FusionDropout'])         

        if len(self._input_slices) != len(self._lstm_specification):
            raise RuntimeError('InputSlices and LSTM config mismatched!')
        
        self._fusion_input_dims = []

        self._embedding = []
        self._lstm = []
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
                    self._embedding.append( ScalarGroupEmbedding( slice_size, embedding_descriptor, 0.0 ) )
                    slice_size = embedding_descriptor[-1]
                else:
                    raise RuntimeError('Unknown embedding type:'+embedding_type)

            self._lstm.append( TkStackedLSTM( slice_size, self._prior_steps_count, self._lstm_specification[i]) )
            self._mlp_input_size = self._mlp_input_size + self._lstm_specification[i][-1]
            self._fusion_input_dims.append( self._lstm_specification[i][-1] )

        self._embedding = torch.nn.ModuleList( self._embedding )
        self._lstm = torch.nn.ModuleList( self._lstm )

        self._fusion = MultiHeadFusionGRF( 
            input_dims=self._fusion_input_dims, 
            embed_dim=self._fusion_embedding_dims, 
            num_heads=self._fusion_attention_heads, 
            dropout=self._fusion_dropout, 
            pooling="mean"
        )

        # enforce gate bias
        with torch.no_grad():
            self._fusion.gate[0].bias.fill_(-2.0)

        # learnable normalization temperature
        self._y_scale = torch.nn.Parameter(torch.tensor(5.0))
        
        self._lstm_output_tensors = None

    def get_trainable_parameters(self, embedding_weight_decay:float, lstm_weight_decay:float, fusion_weight_decay:float, mlp_weight_decay:float, embedding_learning_rate:float, lstm_learning_rate:float, fusion_learning_rate:float, mlp_learning_rate:float):

        embedding_decay_params = []
        embedding_no_decay_params = []

        lstm_decay_params = self._lstm.parameters()
        lstm_no_decay_params = []
        
        mlp_decay_params = []
        mlp_no_decay_params = []

        fusion_decay_params = []
        fusion_no_decay_params = []

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

        for name, param in self._aux_mlp.named_parameters():
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

        other = [ self._y_scale ]

        return [
            {"params": embedding_decay_params, "weight_decay": embedding_weight_decay, 'lr': embedding_learning_rate},
            {"params": embedding_no_decay_params, "weight_decay": 0.0, 'lr': embedding_learning_rate},
            {"params": lstm_decay_params, "weight_decay": lstm_weight_decay, 'lr': lstm_learning_rate},
            {"params": lstm_no_decay_params, "weight_decay": 0.0, 'lr': lstm_learning_rate},
            {"params": mlp_decay_params, "weight_decay": mlp_weight_decay, 'lr': mlp_learning_rate},
            {"params": mlp_no_decay_params, "weight_decay": 0.0, 'lr': mlp_learning_rate},
            {"params": fusion_decay_params, "weight_decay": fusion_weight_decay, 'lr': fusion_learning_rate},
            {"params": fusion_no_decay_params, "weight_decay": 0.0, 'lr': fusion_learning_rate},
            {"params": other, "weight_decay": 0.0, 'lr': mlp_learning_rate},
        ]

    def fusion_group_indices(self):
        return [6,7]
    
    def fusion_decay_group_indices(self):
        return [6]

    def lstm_output(self):
        return self._lstm_output_tensors

    def input_slice(self, idx:int):
        return self._input_slice_tensors[idx]

    def forward(self, input):

        batch_size = input.shape[0]

        input = torch.reshape( input, ( batch_size, self._prior_steps_count, self._input_width) )

        self._input_slice_tensors = []
        self._lstm_output_tensors = []

        for i in range(len(self._input_slices)):
            ch0 = self._input_slices[i][0]
            ch1 = self._input_slices[i][1]
            slice_size = ch1 - ch0
            input_slice_tensor = input[:, :, ch0:ch1]
            input_slice_tensor = self._embedding[i]( input_slice_tensor )
            self._input_slice_tensors.append( input_slice_tensor )
            lstm_output = self._lstm[i]( input_slice_tensor )
            lstm_output = torch.nn.functional.layer_norm( lstm_output, [lstm_output.shape[1]] )
            self._lstm_output_tensors.append( lstm_output )
        
        fused, source_attn, gates = self._fusion(self._lstm_output_tensors)

        fused = torch.nn.functional.layer_norm( fused, [fused.shape[1]] )
        self._lstm_output_tensors.append( fused )
        merged = torch.cat( self._lstm_output_tensors, dim=-1)

        y = self._mlp.forward( merged )
        y = torch.reshape( y, (y.shape[0],y.shape[1]*y.shape[2]))
        y = y / self._y_scale.clamp(2.0, 10.0)

        # no fusion case
        # y = self._mlp.forward( torch.cat( self._lstm_output_tensors, dim=-1) )

        y_aux = self._aux_mlp.forward( self._lstm_output_tensors[self._aux_loss_slice] )
        y_aux = torch.reshape( y_aux, (y_aux.shape[0],y_aux.shape[1]*y_aux.shape[2]))

        if not self.training:
            y = torch.nn.functional.softmax(y, dim=1)

        y_probs = torch.nn.functional.softmax(y, dim=-1)
        y_entropy = -(y_probs * torch.log(y_probs + 1e-8)).sum(dim=-1)
        
        y_diversity = torch.nn.functional.normalize(y, dim=-1)
        y_diversity = y_diversity @ y_diversity.T

        # monitoring feedback
        self._lstm_output_tensors = merged
        # self._lstm_output_tensors = gates
        # self._lstm_output_tensors = self._lstm_output_tensors[self._display_slice]        

        return y, y_entropy, y_diversity, y_aux

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

        return js.mean()        
    
    @staticmethod
    def emd_1d_from_logits(logits, target):
    
        q = torch.nn.functional.softmax(logits, dim=-1)

        # cumulative distributions
        cdf_q = torch.cumsum(q, dim=-1)
        cdf_t = torch.cumsum(target, dim=-1)

        emd = torch.abs(cdf_q - cdf_t).sum(dim=-1)
        return emd.mean()