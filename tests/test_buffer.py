import numpy as np
import torch

from ssd.buffer import TrajectoryBuffer


def env_style_stacks(frames: np.ndarray, stack: int) -> np.ndarray:
    """What a frame-stacking env returns: [T+1, stack * d], first frame repeated at reset."""
    out = []
    for t in range(len(frames)):
        out.append(np.concatenate([frames[max(t - s, 0)] for s in range(stack - 1, -1, -1)]))
    return np.stack(out)


def test_frame_stacked_buffer_matches_environment_stacks():
    rng = np.random.default_rng(0)
    n, horizon, d, stack = 3, 6, 5, 3
    frames = rng.integers(0, 255, size=(n, horizon + 1, d), dtype=np.uint8)
    obs = np.stack([env_style_stacks(f, stack) for f in frames])  # [n, T+1, stack * d]
    buf = TrajectoryBuffer(capacity=100, horizon=horizon, obs_dim=stack * d, act_dim=2, dim_skill=2, device="cpu",
                           frame_stack=stack, obs_dtype=torch.uint8)
    buf.add(obs, np.zeros((n, horizon, 2)), np.zeros((n, horizon)), np.zeros((n, 2)))
    assert buf.obs.shape == (buf.n_max, horizon + 1, d)  # one frame per step stored

    traj, _ = buf.sample_trajectories(n)
    got = sorted(map(bytes, traj.numpy().reshape(n, -1)))
    assert got == sorted(map(bytes, obs.reshape(n, -1)))  # every stack rebuilt exactly

    batch = buf.sample_transitions(64)
    lookup = {bytes(o): (i, t) for i in range(n) for t, o in enumerate(obs[i])}
    for o, o2 in zip(batch["obs"].numpy(), batch["next_obs"].numpy()):
        i, t = lookup[bytes(o)]
        np.testing.assert_array_equal(o2, obs[i, t + 1])


def test_state_buffer_unchanged():
    buf = TrajectoryBuffer(capacity=40, horizon=4, obs_dim=3, act_dim=1, dim_skill=2, device="cpu")
    obs = np.arange(2 * 5 * 3, dtype=np.float32).reshape(2, 5, 3)
    buf.add(obs, np.zeros((2, 4, 1)), np.zeros((2, 4)), np.ones((2, 2)))
    b = buf.sample_transitions(16)
    np.testing.assert_array_equal(b["next_obs"] - b["obs"], np.full((16, 3), 3.0))
