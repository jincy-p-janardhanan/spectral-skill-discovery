import math

import torch
from torch import nn
from torch.nn import functional as F


def mlp(in_dim: int, out_dim: int, hidden_dim: int, n_layers: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for _ in range(n_layers):
        layers += [nn.Linear(d, hidden_dim), nn.ReLU()]
        d = hidden_dim
    layers.append(nn.Linear(d, out_dim))
    net = nn.Sequential(*layers)
    # same init as METRA
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    return net


class Encoder(nn.Module):
    """phi_theta: S -> R^K. Deterministic, as in METRA's inner-product parameterisation."""

    def __init__(self, obs_dim: int, dim_skill: int, hidden_dim: int, n_layers: int):
        super().__init__()
        self.net = mlp(obs_dim, dim_skill, hidden_dim, n_layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class TanhGaussianPolicy(nn.Module):
    """pi_omega(a | s, z): tanh-squashed Gaussian, rescaled to the action bounds."""

    LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0

    def __init__(self, obs_dim, dim_skill, act_dim, hidden_dim, n_layers, act_low, act_high):
        super().__init__()
        self.net = mlp(obs_dim + dim_skill, 2 * act_dim, hidden_dim, n_layers)
        low = torch.as_tensor(act_low, dtype=torch.float32)
        high = torch.as_tensor(act_high, dtype=torch.float32)
        self.register_buffer("act_scale", (high - low) / 2)
        self.register_buffer("act_bias", (high + low) / 2)

    def forward(self, obs: torch.Tensor, z: torch.Tensor, deterministic: bool = False):
        """Returns (action, log_prob). log_prob is None when deterministic."""
        mean, log_std = self.net(torch.cat([obs, z], dim=-1)).chunk(2, dim=-1)
        if deterministic:
            return torch.tanh(mean) * self.act_scale + self.act_bias, None
        log_std = log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp()
        u = mean + std * torch.randn_like(mean)
        # log N(u; mean, std) minus the log-det-Jacobian of a = scale * tanh(u) + bias
        log_prob = (-0.5 * ((u - mean) / std).square() - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)
        log_prob = log_prob - (2 * (math.log(2) - u - F.softplus(-2 * u))).sum(-1)
        log_prob = log_prob - torch.log(self.act_scale).sum()
        return torch.tanh(u) * self.act_scale + self.act_bias, log_prob


class DoubleQ(nn.Module):
    """Twin critics Q_psi(s, z, a)."""

    def __init__(self, obs_dim, dim_skill, act_dim, hidden_dim, n_layers):
        super().__init__()
        in_dim = obs_dim + dim_skill + act_dim
        self.q1 = mlp(in_dim, 1, hidden_dim, n_layers)
        self.q2 = mlp(in_dim, 1, hidden_dim, n_layers)

    def forward(self, obs, z, act):
        x = torch.cat([obs, z, act], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)
