"""
CNN + LSTM Actor-Critic policy for the TornadoTrackEnv.

Architecture:
  - CNN encoder: extracts spatial features from each (C, H, W) observation
  - LSTM: aggregates temporal context across timesteps
  - Actor head: outputs Gaussian distribution over [dx, dy, dr]
  - Lifecycle head: sigmoid probability P(tornado on ground)
  - EF classification head (optional): 6-class softmax for EF0–EF5

The Gaussian actor outputs (mean, log_std) enabling confidence interval
polygon generation: sample many trajectories → compute 1σ / 2σ swath polygons.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from config import cfg

_N_CHANNELS = len(cfg.mrms.variables)
_CNN_CHANNELS = cfg.model.cnn_channels
_LSTM_HIDDEN = cfg.model.lstm_hidden
_ACTOR_HIDDEN = cfg.model.actor_hidden
_LIFECYCLE_HIDDEN = cfg.model.lifecycle_hidden
_EF_CLASSES = cfg.model.ef_classes
_EF_ENABLED = cfg.model.ef_classification
_ACTION_DIM = 3  # [dx, dy, dr]


class CNNEncoder(nn.Module):
    """Spatial feature encoder applied independently to each timestep."""

    def __init__(self, in_channels: int, out_channels: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        ch = in_channels
        for out_ch in out_channels:
            layers += [
                nn.Conv2d(ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            ch = out_ch
        self.net = nn.Sequential(*layers)
        self.out_channels = ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, C, H, W) → flat feature vector (batch, features)."""
        out = self.net(x)
        return out.flatten(1)

    @property
    def output_size(self) -> int:
        # Calculate output size after pooling
        size = 200  # cfg.zarr.grid_size
        for _ in _CNN_CHANNELS:
            size = size // 2
        return self.out_channels * (size ** 2)


class TornadoPolicy(nn.Module):
    """
    Full actor-critic policy:
      CNN encoder → LSTM → Actor head + Lifecycle head + (optional) EF head.
    """

    def __init__(self):
        super().__init__()

        self.cnn = CNNEncoder(_N_CHANNELS, _CNN_CHANNELS)
        cnn_out = self.cnn.output_size

        self.lstm = nn.LSTM(
            input_size=cnn_out,
            hidden_size=_LSTM_HIDDEN,
            num_layers=1,
            batch_first=True,
        )

        # Actor head: Gaussian over [dx, dy, dr]
        self.actor_net = nn.Sequential(
            nn.Linear(_LSTM_HIDDEN, _ACTOR_HIDDEN),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(_ACTOR_HIDDEN, _ACTION_DIM)
        self.actor_log_std = nn.Parameter(torch.zeros(_ACTION_DIM))

        # Critic head: scalar value estimate
        self.critic = nn.Sequential(
            nn.Linear(_LSTM_HIDDEN, _ACTOR_HIDDEN),
            nn.Tanh(),
            nn.Linear(_ACTOR_HIDDEN, 1),
        )

        # Lifecycle head: P(tornado on ground)
        self.lifecycle_net = nn.Sequential(
            nn.Linear(_LSTM_HIDDEN, _LIFECYCLE_HIDDEN),
            nn.ReLU(),
            nn.Linear(_LIFECYCLE_HIDDEN, 1),
            nn.Sigmoid(),
        )

        # Optional EF classification head
        self.ef_head: nn.Module | None = None
        if _EF_ENABLED:
            self.ef_head = nn.Sequential(
                nn.Linear(_LSTM_HIDDEN, _LIFECYCLE_HIDDEN),
                nn.ReLU(),
                nn.Linear(_LIFECYCLE_HIDDEN, _EF_CLASSES),
            )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(
        self,
        obs: torch.Tensor,
        lstm_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            obs: (batch, C, H, W) — single timestep observation
            lstm_state: Optional LSTM hidden + cell state from previous step

        Returns dict with keys:
            action_mean, action_log_std, value, lifecycle_prob,
            lstm_h, lstm_c, ef_logits (if enabled)
        """
        batch = obs.shape[0]

        # CNN encode
        cnn_out = self.cnn(obs)  # (batch, cnn_features)

        # LSTM expects (batch, seq, features); seq=1 for online inference
        lstm_in = cnn_out.unsqueeze(1)  # (batch, 1, cnn_features)
        if lstm_state is None:
            lstm_state = (
                torch.zeros(1, batch, _LSTM_HIDDEN, device=obs.device),
                torch.zeros(1, batch, _LSTM_HIDDEN, device=obs.device),
            )
        lstm_out, (h, c) = self.lstm(lstm_in, lstm_state)
        features = lstm_out[:, 0, :]  # (batch, lstm_hidden)

        # Actor
        actor_hidden = self.actor_net(features)
        action_mean = self.actor_mean(actor_hidden)
        action_log_std = self.actor_log_std.expand_as(action_mean)

        # Critic
        value = self.critic(features)

        # Lifecycle
        lifecycle_prob = self.lifecycle_net(features)

        result = {
            "action_mean": action_mean,
            "action_log_std": action_log_std,
            "value": value,
            "lifecycle_prob": lifecycle_prob,
            "lstm_h": h,
            "lstm_c": c,
        }

        if self.ef_head is not None:
            result["ef_logits"] = self.ef_head(features)

        return result

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
        lstm_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        CleanRL-compatible interface.

        Returns: (action, log_prob, entropy, value, aux_dict)
        """
        out = self.forward(obs, lstm_state)
        dist = Normal(out["action_mean"], out["action_log_std"].exp())
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return (
            action,
            log_prob,
            entropy,
            out["value"].squeeze(-1),
            {
                "lifecycle_prob": out["lifecycle_prob"],
                "lstm_h": out["lstm_h"],
                "lstm_c": out["lstm_c"],
                "ef_logits": out.get("ef_logits"),
            },
        )

    def get_confidence_polygons(
        self,
        obs_sequence: torch.Tensor,
        n_samples: int = 200,
        sigma_levels: list[float] | None = None,
    ) -> dict[str, list]:
        """
        Sample trajectory rollouts to build confidence interval polygons.

        Args:
            obs_sequence: (T, C, H, W) observation sequence
            n_samples:    Number of stochastic rollouts
            sigma_levels: Sigma multiples for polygon radii (default: [1.0, 2.0])

        Returns:
            dict with 'mean_trajectory', 'sigma_1_polygon', 'sigma_2_polygon'
            as lists of (lat, lon) tuples.
        """
        sigma_levels = sigma_levels or cfg.inference.confidence_sigma
        self.eval()
        all_positions: list[list[tuple[float, float]]] = []

        with torch.no_grad():
            for _ in range(n_samples):
                positions = []
                h = c = None
                y, x = obs_sequence.shape[-2] / 2, obs_sequence.shape[-1] / 2

                for t in range(obs_sequence.shape[0]):
                    ob = obs_sequence[t].unsqueeze(0)
                    lstm_state = (h, c) if h is not None else None
                    action, _, _, _, aux = self.get_action_and_value(ob, lstm_state=lstm_state)
                    h, c = aux["lstm_h"], aux["lstm_c"]
                    dx, dy = action[0, 0].item(), action[0, 1].item()
                    x = float(np.clip(x + dx, 0, obs_sequence.shape[-1] - 1))
                    y = float(np.clip(y + dy, 0, obs_sequence.shape[-2] - 1))
                    positions.append((y, x))

                all_positions.append(positions)

        positions_arr = np.array(all_positions)  # (n_samples, T, 2)
        mean_traj = positions_arr.mean(axis=0).tolist()
        std_traj = positions_arr.std(axis=0)  # (T, 2)

        return {
            "mean_trajectory": mean_traj,
            "std_trajectory": std_traj.tolist(),
            "sigma_levels": sigma_levels,
        }
