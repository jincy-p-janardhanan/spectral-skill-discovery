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


class PixelCNN(nn.Module):
    """METRA's image encoder. Input: flat uint8 observations of shape pixel_shape = (C, H, W)."""

    def __init__(self, pixel_shape: tuple[int, int, int], depth: int = 48, n_layers: int = 4):
        super().__init__()
        self.pixel_shape = tuple(pixel_shape)
        layers: list[nn.Module] = []
        c = pixel_shape[0]
        for i in range(n_layers):
            layers += [nn.Conv2d(c, depth * 2 ** i, kernel_size=4, stride=2), nn.ELU()]
            c = depth * 2 ** i
        self.net = nn.Sequential(*layers)
        with torch.no_grad():
            self.out_dim = self.net(torch.zeros(1, *pixel_shape)).numel()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs.reshape(-1, *self.pixel_shape).float() / 255.0
        return self.net(x).flatten(1)


def obs_encoder(obs_dim: int, pixel_shape) -> tuple[nn.Module, int]:
    """Identity for state observations, a new PixelCNN for images."""
    if pixel_shape is None:
        return nn.Identity(), obs_dim
    cnn = PixelCNN(pixel_shape)
    return cnn, cnn.out_dim


class Encoder(nn.Module):
    """phi_theta: S -> R^K.

    input_idx restricts phi to part of the state observation. per_dim uses a separate network per dimension.
    """

    def __init__(self, obs_dim: int, dim_skill: int, hidden_dim: int, n_layers: int,
                 input_idx: list[int] | None = None, pixel_shape=None, per_dim: bool = False):
        super().__init__()
        idx = torch.arange(obs_dim) if input_idx is None else torch.as_tensor(input_idx, dtype=torch.long)
        self.register_buffer("input_idx", idx)
        self.is_pixel = pixel_shape is not None
        self.per_dim = per_dim
        if per_dim:
            self.dims = nn.ModuleList()
            for _ in range(dim_skill):
                enc, feat = obs_encoder(len(idx), pixel_shape)
                self.dims.append(nn.Sequential(enc, mlp(feat, 1, hidden_dim, n_layers)))
        else:
            self.obs_enc, feat = obs_encoder(len(idx), pixel_shape)
            self.net = mlp(feat, dim_skill, hidden_dim, n_layers)

    def _inputs(self, obs: torch.Tensor) -> torch.Tensor:
        return obs if self.is_pixel else obs.index_select(-1, self.input_idx)

    def features(self, obs: torch.Tensor) -> torch.Tensor:
        """Everything below the final linear layer."""
        assert not self.per_dim, "a per-dimension encoder has no shared trunk"
        return self.net[:-1](self.obs_enc(self._inputs(obs)))

    def head(self, features: torch.Tensor) -> torch.Tensor:
        return self.net[-1](features)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.per_dim:
            x = self._inputs(obs)
            return torch.cat([net(x) for net in self.dims], dim=-1)
        return self.head(self.features(obs))


class TanhGaussianPolicy(nn.Module):
    """pi_omega(a | s, z): tanh-squashed Gaussian, rescaled to the action bounds."""

    LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0

    def __init__(self, obs_dim, dim_skill, act_dim, hidden_dim, n_layers, act_low, act_high, pixel_shape=None):
        super().__init__()
        self.obs_enc, feat = obs_encoder(obs_dim, pixel_shape)
        self.net = mlp(feat + dim_skill, 2 * act_dim, hidden_dim, n_layers)
        low = torch.as_tensor(act_low, dtype=torch.float32)
        high = torch.as_tensor(act_high, dtype=torch.float32)
        self.register_buffer("act_scale", (high - low) / 2)
        self.register_buffer("act_bias", (high + low) / 2)

    def forward(self, obs: torch.Tensor, z: torch.Tensor, deterministic: bool = False):
        """Returns (action, log_prob). log_prob is None when deterministic."""
        mean, log_std = self.net(torch.cat([self.obs_enc(obs), z], dim=-1)).chunk(2, dim=-1)
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
    """Twin critics Q(s, z, a), each with its own observation encoder."""

    def __init__(self, obs_dim, dim_skill, act_dim, hidden_dim, n_layers, pixel_shape=None):
        super().__init__()
        self.enc1, feat = obs_encoder(obs_dim, pixel_shape)
        self.enc2, _ = obs_encoder(obs_dim, pixel_shape)
        self.q1 = mlp(feat + dim_skill + act_dim, 1, hidden_dim, n_layers)
        self.q2 = mlp(feat + dim_skill + act_dim, 1, hidden_dim, n_layers)

    def forward(self, obs, z, act):
        x1 = torch.cat([self.enc1(obs), z, act], dim=-1)
        x2 = torch.cat([self.enc2(obs), z, act], dim=-1)
        return self.q1(x1).squeeze(-1), self.q2(x2).squeeze(-1)
