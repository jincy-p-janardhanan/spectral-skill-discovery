import torch


class TrajectoryBuffer:
    """Replay buffer D of fixed-horizon trajectories (s_0..s_T, a_0..a_{T-1}, z).

    Whole trajectories are kept so the spectrum readout can sample
    (s_0, s_T, z) triples; transitions are sampled uniformly for the updates. Stored on `device`.
    """

    def __init__(self, capacity: int, horizon: int, obs_dim: int, act_dim: int, dim_skill: int, device):
        self.n_max = max(1, capacity // horizon)
        self.horizon = horizon
        self.device = device
        self.obs = torch.zeros(self.n_max, horizon + 1, obs_dim, device=device)
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
        self.obs[idx] = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        self.act[idx] = torch.as_tensor(act, dtype=torch.float32, device=self.device)
        self.done[idx] = torch.as_tensor(done, dtype=torch.float32, device=self.device)
        self.z[idx] = torch.as_tensor(z, dtype=torch.float32, device=self.device)
        self.ptr = (self.ptr + n) % self.n_max
        self.size = min(self.size + n, self.n_max)

    def sample_transitions(self, batch_size: int) -> dict[str, torch.Tensor]:
        i = torch.randint(self.size, (batch_size,), device=self.device)
        t = torch.randint(self.horizon, (batch_size,), device=self.device)
        return {
            "obs": self.obs[i, t],
            "next_obs": self.obs[i, t + 1],
            "act": self.act[i, t],
            "done": self.done[i, t],
            "z": self.z[i],
        }

    def sample_trajectories(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = torch.randperm(self.size, device=self.device)[:n]
        return self.obs[i], self.z[i]
