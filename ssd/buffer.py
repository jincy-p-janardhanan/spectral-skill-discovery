import torch


class TrajectoryBuffer:
    """Replay buffer of fixed-horizon trajectories (s_0..s_T, a_0..a_{T-1}, z).

    With frame_stack > 1 only the newest frame of each observation is stored; stacks are rebuilt on sampling.
    """

    def __init__(self, capacity: int, horizon: int, obs_dim: int, act_dim: int, dim_skill: int, device,
                 frame_stack: int = 1, obs_dtype: torch.dtype = torch.float32):
        self.n_max = max(1, capacity // horizon)
        self.horizon = horizon
        self.device = device
        self.frame_stack = frame_stack
        self.frame_dim = obs_dim // frame_stack
        self.obs = torch.zeros(self.n_max, horizon + 1, self.frame_dim, dtype=obs_dtype, device=device)
        self.act = torch.zeros(self.n_max, horizon, act_dim, device=device)
        self.done = torch.zeros(self.n_max, horizon, device=device)
        self.z = torch.zeros(self.n_max, dim_skill, device=device)
        self.ptr = 0
        self.size = 0

    @property
    def n_transitions(self) -> int:
        return self.size * self.horizon

    def add(self, obs, act, done, z):
        """Add N trajectories: obs [N, T+1, d], act [N, T, a], done [N, T], z [N, K]."""
        n = obs.shape[0]
        idx = (self.ptr + torch.arange(n)) % self.n_max
        obs = torch.as_tensor(obs, device=self.device)[..., -self.frame_dim:]  # newest frame only
        self.obs[idx] = obs.to(self.obs.dtype)
        self.act[idx] = torch.as_tensor(act, dtype=torch.float32, device=self.device)
        self.done[idx] = torch.as_tensor(done, dtype=torch.float32, device=self.device)
        self.z[idx] = torch.as_tensor(z, dtype=torch.float32, device=self.device)
        self.ptr = (self.ptr + n) % self.n_max
        self.size = min(self.size + n, self.n_max)

    def _stacked(self, i: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Stacked observations at (i, t), repeating frame 0 before the episode start."""
        if self.frame_stack == 1:
            return self.obs[i, t]
        offsets = torch.arange(1 - self.frame_stack, 1, device=self.device)
        steps = (t.unsqueeze(-1) + offsets).clamp(min=0)
        return self.obs[i.unsqueeze(-1), steps].flatten(-2)

    def sample_transitions(self, batch_size: int) -> dict[str, torch.Tensor]:
        i = torch.randint(self.size, (batch_size,), device=self.device)
        t = torch.randint(self.horizon, (batch_size,), device=self.device)
        return {
            "obs": self._stacked(i, t),
            "next_obs": self._stacked(i, t + 1),
            "act": self.act[i, t],
            "done": self.done[i, t],
            "z": self.z[i],
        }

    def sample_trajectories(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = torch.randperm(self.size, device=self.device)[:n]
        t = torch.arange(self.horizon + 1, device=self.device)
        return self._stacked(i.unsqueeze(-1).expand(-1, len(t)), t.expand(len(i), -1)), self.z[i]

    def state_dict(self) -> dict:
        n = self.size
        return {"obs": self.obs[:n].cpu(), "act": self.act[:n].cpu(), "done": self.done[:n].cpu(),
                "z": self.z[:n].cpu(), "ptr": self.ptr, "size": self.size}

    def load_state_dict(self, state: dict):
        n = state["size"]
        assert n <= self.n_max, f"saved buffer holds {n} trajectories, capacity is {self.n_max}"
        self.obs[:n] = state["obs"].to(self.device)
        self.act[:n] = state["act"].to(self.device)
        self.done[:n] = state["done"].to(self.device)
        self.z[:n] = state["z"].to(self.device)
        self.ptr, self.size = state["ptr"], n
