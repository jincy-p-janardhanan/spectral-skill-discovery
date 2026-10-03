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
    discrete: bool = False             # one-hot skills with METRA's centred mask
    signed_skills: bool = False        # discrete skills z = +-e_k
    cube_skills: bool = False          # discrete skills z in {-1, 1}^K / sqrt(K)
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
    temporal_per_dim: bool = False     # mean_k (dphi_k)^2 <= 1 as in METRA's code
    decor_warmup: int = 0              # updates before decorrelation starts
    decor_ramp: int = 0                # updates over which the decorrelation weight ramps to 1
    decor_head_only: bool = False      # decorrelation gradients only reach the last layer of phi
    decor_scale: float = 0.0           # > 0: weight C_kj by s_k s_j / (s_k s_j + decor_scale)
    encoder_per_dim: bool = False      # separate encoder network per dimension
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
                 encoder_input_idx: list[int] | None = None, pixel_shape=None):
        self.cfg = cfg
        self.device = device
        k = cfg.dim_skill

        self.encoder = Encoder(obs_dim, k, cfg.hidden_dim, cfg.n_layers, encoder_input_idx, pixel_shape,
                               per_dim=cfg.encoder_per_dim).to(device)
        self.target_encoder = copy.deepcopy(self.encoder).requires_grad_(False)
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=cfg.lr_theta)

        self.sac = SAC(
            obs_dim=obs_dim, dim_skill=k, act_dim=act_dim, act_low=act_low, act_high=act_high,
            hidden_dim=cfg.hidden_dim, n_layers=cfg.n_layers,
            lr_actor=cfg.lr_actor, lr_critic=cfg.lr_critic, lr_alpha=cfg.lr_alpha,
            alpha_init=cfg.alpha_init, tau=cfg.sac_tau, gamma=cfg.gamma,
            target_coef=cfg.target_coef, device=device, pixel_shape=pixel_shape,
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

    def _decor_weight(self, phi_s: torch.Tensor) -> torch.Tensor | float:
        """Per-pair weight s_k s_j / (s_k s_j + decor_scale) on C_kj, or 1 if decor_scale is 0."""
        if self.cfg.decor_scale <= 0:
            return 1.0
        s = phi_s.detach().std(dim=0, correction=0)
        ss = s[self.rows] * s[self.cols]
        return ss / (ss + self.cfg.decor_scale)

    def _constraints(self, phi_s: torch.Tensor, phi_next: torch.Tensor):
        """Returns (weighted C, h, unweighted C)."""
        c = spectral.decorrelation(phi_s, self.rows, self.cols, self.cfg.eps, self.cfg.sigma_k_grad)
        h = spectral.temporal_hinge(phi_s, phi_next, self.cfg.temporal_per_dim)
        return c * self._decor_weight(phi_s), h, c

    def _encode_pair(self, encoder, batch):
        phi = encoder(torch.cat([batch["obs"], batch["next_obs"]], dim=0))
        return phi.chunk(2, dim=0)

    def skill_weights(self, z: torch.Tensor) -> torch.Tensor:
        """Vector the phi differences are projected on: z, or (z - mean(z)) * K / (K - 1) for one-hot skills."""
        if not self.cfg.discrete or self.cfg.signed_skills or self.cfg.cube_skills:
            return z
        k = z.shape[-1]
        return (z - z.mean(dim=-1, keepdim=True)) * k / max(k - 1, 1)

    def update(self, buffer: TrajectoryBuffer) -> dict[str, torch.Tensor]:
        cfg = self.cfg

        # policy update
        b = buffer.sample_transitions(cfg.batch_size)
        with torch.no_grad():
            reward_enc = self.target_encoder if cfg.reward_use_target else self.encoder
            reward = spectral.skill_objective(*self._encode_pair(reward_enc, b), self.skill_weights(b["z"]))
        metrics = self.sac.update(b["obs"], b["act"], b["next_obs"], b["z"], reward * cfg.scale_reward, b["done"])

        # encoder update on two independent batches
        b1 = buffer.sample_transitions(cfg.batch_size)
        b2 = buffer.sample_transitions(cfg.batch_size)
        pair1 = torch.cat([b1["obs"], b1["next_obs"]], dim=0)
        if cfg.decor_head_only:
            feat1 = self.encoder.features(pair1)
            phi1_s, phi1_next = self.encoder.head(feat1).chunk(2, dim=0)
        else:
            phi1_s, phi1_next = self.encoder(pair1).chunk(2, dim=0)
        c1, h1, c1_raw = self._constraints(phi1_s, phi1_next)
        if cfg.decor_head_only:
            c1 = spectral.decorrelation(self.encoder.head(feat1.detach()).chunk(2, dim=0)[0], self.rows, self.cols,
                                        cfg.eps, cfg.sigma_k_grad)
        with torch.no_grad():
            c2, h2, c2_raw = self._constraints(*self._encode_pair(self.encoder, b2))
        j_hat = spectral.skill_objective(phi1_s, phi1_next, self.skill_weights(b1["z"])).mean()

        # decorrelation weight: 0 during the warm-up, then a linear ramp to 1
        delta = float(self.n_updates >= cfg.decor_warmup)
        if cfg.decor_ramp > 0:
            delta = min(1.0, max(0.0, (self.n_updates - cfg.decor_warmup + 1) / cfg.decor_ramp))
        decor_on = delta > 0
        lagrangian = spectral.lagrangian(j_hat, c1 * delta, h1, c2, h2, self.beta, self.nu, self.c)
        self.encoder_opt.zero_grad(set_to_none=True)
        (-lagrangian).backward()  # gradient ascent on L
        self.encoder_opt.step()

        # duals
        self.n_updates += 1
        c1, h1 = c1.detach(), h1.detach()
        if self.n_updates % cfg.n_dual == 0:
            if decor_on:
                self.beta += cfg.eta_beta * delta * 0.5 * (c1 + c2)
            self.nu = torch.clamp(self.nu + cfg.eta_nu * 0.5 * (h1 + h2), min=0.0)

        # barrier and EMA target
        if decor_on:
            self.c = min(cfg.c_max, self.c * (1 + cfg.kappa))
        with torch.no_grad():
            for tp, p in zip(self.target_encoder.parameters(), self.encoder.parameters()):
                tp.lerp_(p, cfg.encoder_tau)

        corr = 0.5 * (c1_raw.detach() + c2_raw)
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
        flat = obs.reshape(n * t1, d)
        phi = torch.cat([enc(chunk) for chunk in flat.split(2048)]).reshape(n, t1, -1)
        return spectral.spectrum_readout(phi, self.skill_weights(z), cfg.n_shuffles, cfg.readout_quantile,
                                         self.readout_gen)

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
            "readout_gen": self.readout_gen.get_state(),
        }

    def load_state_dict(self, state):
        self.encoder.load_state_dict(state["encoder"])
        self.target_encoder.load_state_dict(state["target_encoder"])
        self.encoder_opt.load_state_dict(state["encoder_opt"])
        self.sac.load_state_dict(state["sac"])
        self.beta = state["beta"].to(self.device)
        self.nu = state["nu"].to(self.device)
        self.c = state["c"]
        self.n_updates = state["n_updates"]
        if "readout_gen" in state:
            self.readout_gen.set_state(state["readout_gen"])
