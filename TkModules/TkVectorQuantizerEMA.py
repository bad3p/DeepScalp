import configparser
import torch
import json
from TkModules.TkModel import TkModel

#------------------------------------------------------------------------------------------------------------------------
# VQ-VAE Core
#------------------------------------------------------------------------------------------------------------------------

class TkVectorQuantizerEMA(torch.nn.Module):
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

        # Reserve index 0
        with torch.no_grad():
            self.embedding.weight.data[0].zero_()

        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_w", self.embedding.weight.data.clone())

        self.random_restart_enabled = False

    def set_random_restart(self, flag:bool):
        self.random_restart_enabled = flag

    def quantize(self, z):
        # z: (B, C, T)
        z = z.permute(0, 2, 1).contiguous()  # (B, T, C)
        flat_z = z.view(-1, self.embedding_dim)  # (B*T, C)

        # 1. Normalize z and the embeddings to compute Cosine Similarity
        flat_z_norm = torch.nn.functional.normalize(flat_z, p=2, dim=1, eps=1e-6)
        weight_norm = torch.nn.functional.normalize(self.embedding.weight, p=2, dim=1, eps=1e-6)

        # 2. Compute Cosine Distance (1 - Cosine Similarity)
        cos_sim = flat_z_norm @ weight_norm.t()
        distances = 1.0 - cos_sim

        # Prevent index 0 from being selected
        distances[:, 0] = float("inf")

        encoding_indices = torch.argmin(distances, dim=1)
        encodings = torch.nn.functional.one_hot(
            encoding_indices, self.num_embeddings
        ).type(flat_z.dtype)

        quantized = self.embedding(encoding_indices).view(z.shape)

        # EMA updates (training only)
        if self.training:
            self.ema_cluster_size.mul_(self.decay).add_(
                encodings.sum(0), alpha=1 - self.decay
            )

            dw = encodings.t() @ flat_z_norm
            self.ema_w.mul_(self.decay).add_(dw, alpha=1 - self.decay)

            # Prevent EMA updates for index 0
            self.ema_cluster_size[0] = 0
            self.ema_w[0] = 0

            n = self.ema_cluster_size.sum()
            cluster_size = (
                (self.ema_cluster_size + self.eps)
                / (n + self.num_embeddings * self.eps)
                * n
            )

            self.embedding.weight.data.copy_(
                self.ema_w / cluster_size.unsqueeze(1)
            )

            # Project updated embeddings back to the unit hypersphere
            self.embedding.weight.data = torch.nn.functional.normalize(
                self.embedding.weight.data, p=2, dim=1, eps=1e-6
            )

            # Keep embedding 0 fixed as exact zero
            self.embedding.weight.data[0].zero_()

            # -----------------------------------------------------------------
            # 3. Dead Code Random Restart Mechanism
            # -----------------------------------------------------------------
            # Check cluster usage. We consider a code "dead" if its EMA usage falls below 1.0.
            # We slice [1:] to ignore index 0, which is intentionally dead.
            if self.random_restart_enabled:
                dead_mask = self.ema_cluster_size[1:] < 1.0
            
                # Get the actual indices of the dead codes (adding 1 back due to the slice)
                dead_indices = torch.nonzero(dead_mask).squeeze(-1) + 1
            
                num_dead = dead_indices.numel()
                if num_dead > 0:
                    # Randomly sample 'num_dead' latents from the current batch's normalized z
                    batch_size = flat_z_norm.shape[0]
                    rand_indices = torch.randint(0, batch_size, (num_dead,), device=z.device)
                    sampled_latents = flat_z_norm[rand_indices]
                
                    # Overwrite the dead embeddings with the fresh latents
                    self.embedding.weight.data[dead_indices] = sampled_latents
                
                    # Reset EMA trackers so the new codes survive the next EMA decay step.
                    # Initialize cluster size to 1.0 to reflect 1 "pseudo-hit".
                    self.ema_cluster_size[dead_indices] = 1.0
                    self.ema_w[dead_indices] = sampled_latents * 1.0


        # commitment loss
        loss = self.commitment_cost * torch.nn.functional.mse_loss(
            quantized.detach(), z
        )

        # straight-through estimator
        quantized = z + (quantized - z).detach()
        quantized = quantized.permute(0, 2, 1).contiguous()

        codes = encoding_indices.view(z.shape[0], z.shape[1])  # (B, T)

        return quantized, loss, codes

    def forward(self, z):
        z_q, loss, codes = self.quantize(z)
        return z_q, loss, codes
