# Spectral Skill Discovery agent: encoder and policy updates, duals, and the spectrum readout.

import copy
from dataclasses import dataclass

import torch

from ssd import spectral
from ssd.buffer import TrajectoryBuffer
from ssd.networks import Encoder
from ssd.sac import SAC


@dataclass
class SSDConfig:
    dim_skill: int = 8                 # K
    hidden_dim: int = 1024
    n_layers: int = 2
    batch_size: int = 256
    # rates
    lr_theta: float = 1e-4
    lr_actor: float = 1e-4
    lr_critic: float = 1e-4
    lr_alpha: float = 1e-4
    eta_beta: float = 1e-2
    eta_nu: float = 1e-2
    # duals and barrier
    nu0: float = 30.0
    c0: float = 1.0
    kappa: float = 1e-5
    c_max: float = 100.0
    n_dual: int = 10                   # dual update period, in updates
    eps: float = 1e-6
    sigma_k_grad: bool = True          # gradient through sigma_k in C_kj
    # EMA target encoder
    encoder_tau: float = 5e-3
    reward_use_target: bool = False
    readout_use_target: bool = False
    # SAC (METRA defaults)
    alpha_init: float = 0.01
    sac_tau: float = 5e-3
    gamma: float = 0.99
    target_coef: float = 1.0
    scale_reward: float = 1.0
    # spectrum readout
    n_readout_trajs: int = 256         # N
    n_shuffles: int = 1000             # M
    readout_quantile: float = 0.95


class SpectralSkillDiscovery:
    def __init__(self, cfg: SSDConfig, obs_dim: int, act_dim: int, act_low, act_high, device, seed: int = 0,
                 encoder_input_idx: list[int] | None = None):
        self.cfg = cfg
        self.device = device
        k = cfg.dim_skill

        self.encoder = Encoder(obs_dim, k, cfg.hidden_dim, cfg.n_layers, encoder_input_idx).to(device)
        self.target_encoder = copy.deepcopy(self.encoder).requires_grad_(False)
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=cfg.lr_theta)

        self.sac = SAC(
            obs_dim=obs_dim, dim_skill=k, act_dim=act_dim, act_low=act_low, act_high=act_high,
            hidden_dim=cfg.hidden_dim, n_layers=cfg.n_layers,
            lr_actor=cfg.lr_actor, lr_critic=cfg.lr_critic, lr_alpha=cfg.lr_alpha,
            alpha_init=cfg.alpha_init, tau=cfg.sac_tau, gamma=cfg.gamma,
            target_coef=cfg.target_coef, device=device,
        )

        # beta is stored flat over the pairs j < k
        self.rows, self.cols = spectral.lower_pairs(k, device=device)
        self.beta = torch.zeros(len(self.rows), device=device)
        self.nu = torch.tensor(cfg.nu0, device=device)
        self.c = cfg.c0
        self.n_updates = 0

        self.readout_gen = torch.Generator(device=device).manual_seed(seed)

    @property
    def policy(self):
        return self.sac.policy

    @torch.no_grad()
    def act(self, obs: torch.Tensor, z: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        return self.policy(obs, z, deterministic=deterministic)[0]

    def _constraints(self, phi_s: torch.Tensor, phi_next: torch.Tensor):
        c = spectral.decorrelation(phi_s, self.rows, self.cols, self.cfg.eps, self.cfg.sigma_k_grad)
        h = spectral.temporal_hinge(phi_s, phi_next)
        return c, h

    def _encode_pair(self, encoder, batch):
        phi = encoder(torch.cat([batch["obs"], batch["next_obs"]], dim=0))
        return phi.chunk(2, dim=0)

    def update(self, buffer: TrajectoryBuffer) -> dict[str, torch.Tensor]:
        cfg = self.cfg

        # policy update
        b = buffer.sample_transitions(cfg.batch_size)
        with torch.no_grad():
            reward_enc = self.target_encoder if cfg.reward_use_target else self.encoder
            reward = spectral.skill_objective(*self._encode_pair(reward_enc, b), b["z"])
        metrics = self.sac.update(b["obs"], b["act"], b["next_obs"], b["z"], reward * cfg.scale_reward, b["done"])

        # encoder update on two independent batches
        b1 = buffer.sample_transitions(cfg.batch_size)
        b2 = buffer.sample_transitions(cfg.batch_size)
        phi1_s, phi1_next = self._encode_pair(self.encoder, b1)
        c1, h1 = self._constraints(phi1_s, phi1_next)
        with torch.no_grad():
            c2, h2 = self._constraints(*self._encode_pair(self.encoder, b2))
        j_hat = spectral.skill_objective(phi1_s, phi1_next, b1["z"]).mean()

        lagrangian = spectral.lagrangian(j_hat, c1, h1, c2, h2, self.beta, self.nu, self.c)
        self.encoder_opt.zero_grad(set_to_none=True)
        (-lagrangian).backward()  # gradient ascent on L
        self.encoder_opt.step()

        # duals
        self.n_updates += 1
        c1, h1 = c1.detach(), h1.detach()
        if self.n_updates % cfg.n_dual == 0:
            self.beta += cfg.eta_beta * 0.5 * (c1 + c2)
            self.nu = torch.clamp(self.nu + cfg.eta_nu * 0.5 * (h1 + h2), min=0.0)

        # barrier and EMA target
        self.c = min(cfg.c_max, self.c * (1 + cfg.kappa))
        with torch.no_grad():
            for tp, p in zip(self.target_encoder.parameters(), self.encoder.parameters()):
                tp.lerp_(p, cfg.encoder_tau)

        corr = 0.5 * (c1 + c2)
        metrics.update({
            "ssd/lagrangian": lagrangian.detach(),
            "ssd/J": j_hat.detach(),
            "ssd/h": 0.5 * (h1 + h2),
            "ssd/C_abs_mean": corr.abs().mean() if len(corr) else torch.zeros((), device=self.device),
            "ssd/C_abs_max": corr.abs().max() if len(corr) else torch.zeros((), device=self.device),
            "ssd/beta_abs_mean": self.beta.abs().mean() if len(self.beta) else torch.zeros((), device=self.device),
            "ssd/nu": self.nu,
            "ssd/intrinsic_reward": reward.mean(),
        })
        return metrics

    @torch.no_grad()
    def spectrum(self, buffer: TrajectoryBuffer) -> spectral.Spectrum:
        """Spectrum readout on trajectories sampled from the buffer."""
        cfg = self.cfg
        obs, z = buffer.sample_trajectories(cfg.n_readout_trajs)
        enc = self.target_encoder if cfg.readout_use_target else self.encoder
        n, t1, d = obs.shape
        phi = enc(obs.reshape(n * t1, d)).reshape(n, t1, -1)
        return spectral.spectrum_readout(phi, z, cfg.n_shuffles, cfg.readout_quantile, self.readout_gen)

    def state_dict(self):
        return {
            "encoder": self.encoder.state_dict(),
            "target_encoder": self.target_encoder.state_dict(),
            "encoder_opt": self.encoder_opt.state_dict(),
            "sac": self.sac.state_dict(),
            "beta": self.beta.cpu(),
            "nu": self.nu.cpu(),
            "c": self.c,
            "n_updates": self.n_updates,
        }
