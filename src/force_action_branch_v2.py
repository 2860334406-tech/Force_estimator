"""Inference interface for the action-aware, force-supervised Route B VAE.

Input: SuperMimic post-step policy observation [batch, 930].
Output: same-step applied torso-frame 3D external force [batch, 3], newtons.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from route_b_vae_full930 import ProprioVAE


class ForceEstimator(nn.Module):
    def __init__(self, state: dict[str, torch.Tensor]):
        super().__init__()
        latent_dim = state["vae.mu.weight"].shape[0]
        self.vae = ProprioVAE(state["vae.x_mean"], state["vae.x_std"],
                              state["vae.next_mean"], state["vae.next_std"], latent_dim)
        self.action_encoder = nn.Sequential(
            nn.Linear(290, 64), nn.ELU(), nn.Linear(64, 16), nn.ELU()
        )
        self.head = nn.Sequential(
            nn.Linear(latent_dim + 16, 64), nn.ELU(),
            nn.Linear(64, 64), nn.ELU(), nn.Linear(64, 3)
        )
        self.load_state_dict(state, strict=True)

    def forward(self, policy_obs: torch.Tensor) -> torch.Tensor:
        if policy_obs.ndim != 2 or policy_obs.shape[1] != 930:
            raise ValueError(f"Expected [batch, 930] policy observation; got {tuple(policy_obs.shape)}")
        x = policy_obs.float()
        mu, _ = self.vae.encode(x)
        action = (x[:, 640:] - self.vae.x_mean[640:]) / self.vae.x_std[640:]
        return 30.0 * self.head(torch.cat((mu, self.action_encoder(action)), dim=-1))


def load_action_force_estimator(
    checkpoint: str | Path,
    *,
    device: str | torch.device = "cuda:0",
) -> ForceEstimator:
    checkpoint = Path(checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    return ForceEstimator(saved["model"]).to(device).eval().requires_grad_(False)
