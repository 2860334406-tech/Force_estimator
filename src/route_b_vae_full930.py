"""Route B: full policy observation VAE. No force or contact labels are read."""
from __future__ import annotations

import torch
from torch import nn


class ProprioVAE(nn.Module):
    def __init__(self, x_mean, x_std, next_mean, next_std, latent_dim=7):
        super().__init__()
        self.register_buffer("x_mean", torch.as_tensor(x_mean, dtype=torch.float32))
        self.register_buffer("x_std", torch.as_tensor(x_std, dtype=torch.float32).clamp_min(1e-6))
        self.register_buffer("next_mean", torch.as_tensor(next_mean, dtype=torch.float32))
        self.register_buffer("next_std", torch.as_tensor(next_std, dtype=torch.float32).clamp_min(1e-6))
        self.encoder = nn.Sequential(
            nn.Linear(self.x_mean.numel(), 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
        )
        self.mu = nn.Linear(128, latent_dim)
        self.logvar = nn.Linear(128, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ELU(),
            nn.Linear(128, 256), nn.ELU(),
            nn.Linear(256, 64),
        )

    def encode(self, x):
        if x.shape[-1] != self.x_mean.numel():
            raise ValueError(f"Expected {self.x_mean.numel()} input values; got {x.shape[-1]}")
        h = self.encoder((x - self.x_mean) / self.x_std)
        return self.mu(h), self.logvar(h).clamp(-12, 8)

    def forward(self, x, sample=True):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if sample else mu
        return self.decoder(z), mu, logvar

    def target(self, next_proprio64):
        return (next_proprio64 - self.next_mean) / self.next_std


def load_vae(path, device="cuda:0"):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    norm = saved["norm"]
    model = ProprioVAE(norm["x_mean"], norm["x_std"], norm["next_mean"],
                       norm["next_std"], saved["config"]["latent_dim"])
    model.load_state_dict(saved["model"], strict=True)
    model.to(device).eval().requires_grad_(False)
    return model
