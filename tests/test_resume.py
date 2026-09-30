import copy

import numpy as np
import torch

from ssd.algorithm import SpectralSkillDiscovery, SSDConfig
from ssd.buffer import TrajectoryBuffer


def make(seed=0):
    torch.manual_seed(seed)
    cfg = SSDConfig(dim_skill=3, hidden_dim=16, batch_size=32, n_dual=1)
    agent = SpectralSkillDiscovery(cfg, obs_dim=4, act_dim=2, act_low=[-1, -1], act_high=[1, 1], device="cpu")
    buf = TrajectoryBuffer(capacity=100, horizon=5, obs_dim=4, act_dim=2, dim_skill=3, device="cpu")
    return agent, buf


def fill(buf, n, seed):
    g = np.random.default_rng(seed)
    buf.add(g.standard_normal((n, 6, 4)), g.uniform(-1, 1, (n, 5, 2)), np.zeros((n, 5)), g.standard_normal((n, 3)))


def test_agent_and_buffer_resume_continue_identically():
    agent, buf = make()
    fill(buf, 30, 0)  # wraps the 20-trajectory ring
    for _ in range(3):
        agent.update(buf)
    agent_state, buf_state = copy.deepcopy(agent.state_dict()), copy.deepcopy(buf.state_dict())

    agent2, buf2 = make(seed=123)  # different init: everything must come from the snapshot
    agent2.load_state_dict(agent_state)
    buf2.load_state_dict(buf_state)
    assert buf2.size == buf.size and buf2.ptr == buf.ptr
    torch.testing.assert_close(buf2.obs, buf.obs)

    # One more update with identical RNG state must give identical parameters and duals.
    torch.manual_seed(7)
    agent.update(buf)
    torch.manual_seed(7)
    agent2.update(buf2)
    for p1, p2 in zip(agent.encoder.parameters(), agent2.encoder.parameters()):
        torch.testing.assert_close(p1, p2)
    for p1, p2 in zip(agent.sac.policy.parameters(), agent2.sac.policy.parameters()):
        torch.testing.assert_close(p1, p2)
    torch.testing.assert_close(agent.beta, agent2.beta)
    assert agent.c == agent2.c and agent.n_updates == agent2.n_updates
