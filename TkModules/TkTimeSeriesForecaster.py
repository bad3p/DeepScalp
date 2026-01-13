
import configparser
import torch
import json
from TkModules.TkModel import TkModel
from TkModules.TkStackedLSTM import TkStackedLSTM
from TkModules.TkSelfAttention import TkSelfAttention
from TkModules.TkTCNN import TkTCNN

# --------------------------------------------------------------------------------------------------------------
# Fusion module
#  input_dims: list of LSTM/TCNN output dimensions [d1, d2, ..., dN]
#  embed_dim: shared embedding dimension
#  num_heads: number of attention heads
#  pooling: 'mean', 'max', or 'attn'
# --------------------------------------------------------------------------------------------------------------

class MultiHeadFusion(torch.nn.Module):
    def __init__(
        self,
        input_dims,
        embed_dim=128,
        num_heads=4,
        dropout=0.1,
        pooling="mean"
    ):
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.num_sources = len(input_dims)
        self.embed_dim = embed_dim
        self.pooling = pooling

        # Project each input to shared space
        self.projections = torch.nn.ModuleList([
            torch.nn.Linear(d, embed_dim) for d in input_dims
        ])

        # Multi-head self-attention over sources
        self.mha = torch.nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Optional attention pooling
        if pooling == "attn":
            self.pool_attn = torch.nn.Linear(embed_dim, 1)

        self.norm = torch.nn.LayerNorm(embed_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, inputs, attn_mask=None):
        """
        inputs: list of tensors [(B, d1), (B, d2), ..., (B, dN)]
        attn_mask: optional (N, N) mask for source attention
        """

        # 1) Project to shared embedding
        projected = [
            proj(h) for proj, h in zip(self.projections, inputs)
        ]  # list of (B, D)

        # Stack into source sequence
        x = torch.stack(projected, dim=1)  # (B, N, D)

        # 2) Multi-head self-attention
        attn_out, attn_weights = self.mha(
            query=x,
            key=x,
            value=x,
            attn_mask=attn_mask,
            need_weights=True
        )  # attn_out: (B, N, D)

        # Residual + normalization
        x = self.norm(x + self.dropout(attn_out))

        # 3) Pool across sources
        if self.pooling == "mean":
            fused = x.mean(dim=1)

        elif self.pooling == "max":
            fused, _ = x.max(dim=1)

        elif self.pooling == "attn":
            scores = self.pool_attn(x).squeeze(-1)  # (B, N)
            weights = torch.softmax(scores, dim=1)
            fused = torch.sum(x * weights.unsqueeze(-1), dim=1)

        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

        return fused, attn_weights


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

        # Self-attention
        attn_out, attn_weights = self.mha(x, x, x, need_weights=True)

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
# Embedding for latent stape inputs
# --------------------------------------------------------------------------------------------------------------

class EmbeddedLatents(torch.nn.Module):
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
        self._target_code_width = int(_cfg['Autoencoders']['LastTradesAutoencoderCodeLayerSize']) 
        self._input_slices = json.loads(_cfg['TimeSeries']['InputSlices'])
        self._embedding_specification = json.loads(_cfg['TimeSeries']['Embedding'])
        self._log_space_embedding = json.loads(_cfg['TimeSeries']['LogSpaceEmbedding'])
        self._embedding_dropout = json.loads(_cfg['TimeSeries']['EmbeddingDropout'])
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
                self._embedding.append( EmbeddedLatents( in_channels=slice_size, specification=self._embedding_specification[i], dropout = self._embedding_dropout[i] ) )
                slice_size = self._embedding_specification[i][-1]

            self._lstm.append( TkStackedLSTM( slice_size, self._prior_steps_count, self._lstm_specification[i]) )
            self._mlp_input_size = self._mlp_input_size + self._lstm_specification[i][-1]
            self._fusion_input_dims.append( self._lstm_specification[i][1] )

        self._embedding = torch.nn.ModuleList( self._embedding )
        self._lstm = torch.nn.ModuleList( self._lstm )

        self._fusion = MultiHeadFusionGRF( 
            input_dims=self._fusion_input_dims, 
            embed_dim=self._fusion_embedding_dims, 
            num_heads=self._fusion_attention_heads, 
            dropout=self._fusion_dropout, 
            pooling="attn"
        )
        
        self._lstm_output_tensors = None

    def get_trainable_parameters(self, embedding_weight_decay:float, lstm_weight_decay:float, mlp_weight_decay:float):

        embedding_decay_params = []
        embedding_no_decay_params = []

        lstm_decay_params = self._lstm.parameters()
        lstm_no_decay_params = []
        
        mlp_decay_params = []
        mlp_no_decay_params = []

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
                mlp_decay_params.append(param)
            else:
                mlp_no_decay_params.append(param)

        return [
            {"params": embedding_decay_params, "weight_decay": embedding_weight_decay},
            {"params": embedding_no_decay_params, "weight_decay": 0.0},
            {"params": lstm_decay_params, "weight_decay": lstm_weight_decay},
            {"params": lstm_no_decay_params, "weight_decay": 0.0},
            {"params": mlp_decay_params, "weight_decay": mlp_weight_decay},
            {"params": mlp_no_decay_params, "weight_decay": 0.0},
        ]

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
            if self._log_space_embedding[i] > 0:
                input_slice_tensor = torch.log(input_slice_tensor + 1e-8)
            input_slice_tensor = self._embedding[i]( input_slice_tensor )
            self._input_slice_tensors.append( input_slice_tensor )
            self._lstm_output_tensors.append( self._lstm[i]( input_slice_tensor ) )
        
        fused, source_attn, gates = self._fusion(self._lstm_output_tensors)        
        y = self._mlp.forward( fused )
        y = torch.reshape( y, ( batch_size, self._target_code_width) )

        y_aux = self._aux_mlp.forward( self._lstm_output_tensors[self._aux_loss_slice] )

        self._lstm_output_tensors = self._lstm_output_tensors[self._display_slice]

        return y, y_aux