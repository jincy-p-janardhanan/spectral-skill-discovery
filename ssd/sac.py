"""Skill-conditioned SAC, following METRA's implementation."""

import copy
import math

import torch
from torch.nn import functional as F

from ssd.networks import DoubleQ, TanhGaussianPolicy


class SAC:
    def __init__(self, *, obs_dim, dim_skill, act_dim, act_low, act_high, hidden_dim, n_layers,
                 lr_actor, lr_critic, lr_alpha, alpha_init, tau, gamma, target_coef, device, pixel_shape=None):
        self.policy = TanhGaussianPolicy(obs_dim, dim_skill, act_dim, hidden_dim, n_layers, act_low, act_high,
                                         pixel_shape).to(device)
        self.critic = DoubleQ(obs_dim, dim_skill, act_dim, hidden_dim, n_layers, pixel_shape).to(device)
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False)
        self.log_alpha = torch.tensor(math.log(alpha_init), device=device, requires_grad=True)

        self.actor_opt = torch.optim.Adam(self.policy.parameters(), lr=lr_actor)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr_alpha)

        self.tau = tau
        self.gamma = gamma
        # METRA uses half the usual -|A| entropy target.
        self.target_entropy = -act_dim / 2 * target_coef

    def update(self, obs, act, next_obs, z, reward, done) -> dict[str, torch.Tensor]:
        alpha = self.log_alpha.exp().detach()

        # Critic
        with torch.no_grad():
            next_act, next_logp = self.policy(next_obs, z)
            tq1, tq2 = self.critic_target(next_obs, z, next_act)
            target = reward + self.gamma * (1.0 - done) * (torch.min(tq1, tq2) - alpha * next_logp)
        q1, q2 = self.critic(obs, z, act)
        critic_loss = 0.5 * F.mse_loss(q1, target) + 0.5 * F.mse_loss(q2, target)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        # Actor
        new_act, logp = self.policy(obs, z)
        q1_pi, q2_pi = self.critic(obs, z, new_act)
        actor_loss = (alpha * logp - torch.min(q1_pi, q2_pi)).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        # Temperature
        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        with torch.no_grad():
            for tp, p in zip(self.critic_target.parameters(), self.critic.parameters()):
                tp.lerp_(p, self.tau)

        return {
            "sac/critic_loss": critic_loss.detach(),
            "sac/actor_loss": actor_loss.detach(),
            "sac/alpha": alpha,
            "sac/entropy": -logp.detach().mean(),
            "sac/q_mean": q1.detach().mean(),
        }

    def state_dict(self):
        return {
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "alpha_opt": self.alpha_opt.state_dict(),
        }

    def load_state_dict(self, state):
        self.policy.load_state_dict(state["policy"])
        self.critic.load_state_dict(state["critic"])
        self.critic_target.load_state_dict(state["critic_target"])
        with torch.no_grad():
            self.log_alpha.copy_(state["log_alpha"].to(self.log_alpha.device))
        self.actor_opt.load_state_dict(state["actor_opt"])
        self.critic_opt.load_state_dict(state["critic_opt"])
        self.alpha_opt.load_state_dict(state["alpha_opt"])
