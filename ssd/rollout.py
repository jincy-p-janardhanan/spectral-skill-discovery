import numpy as np
import torch


@torch.no_grad()
def rollout(envs, policy, z: np.ndarray, horizon: int, device, deterministic: bool = False) -> dict[str, np.ndarray]:
    """Run len(envs) episodes of `horizon` steps in lockstep, env i following skill z[i]."""
    n = len(envs)
    obs = np.stack([env.reset()[0] for env in envs])
    xy0 = np.stack([env.unwrapped._xy() for env in envs])
    z_t = torch.as_tensor(z, dtype=torch.float32, device=device)

    obs_buf = np.empty((n, horizon + 1, obs.shape[-1]), dtype=np.float32)
    xy_buf = np.empty((n, horizon + 1, 2), dtype=np.float32)
    act_buf = np.empty((n, horizon, envs[0].action_space.shape[0]), dtype=np.float32)
    done_buf = np.zeros((n, horizon), dtype=np.float32)
    obs_buf[:, 0], xy_buf[:, 0] = obs, xy0

    for t in range(horizon):
        obs_t = torch.as_tensor(obs_buf[:, t], device=device)
        act = policy(obs_t, z_t, deterministic=deterministic)[0].cpu().numpy()
        act_buf[:, t] = act
        for i, env in enumerate(envs):
            o, _, terminated, _, info = env.step(act[i])
            obs_buf[i, t + 1] = o
            xy_buf[i, t + 1] = info["next_coordinates"]
            done_buf[i, t] = float(terminated)  # time-limit truncation is not a terminal
    return {"obs": obs_buf, "act": act_buf, "done": done_buf, "z": np.asarray(z, dtype=np.float32), "xy": xy_buf}


def coverage(xy: np.ndarray, env_name: str) -> int:
    """METRA's state-coverage metric: number of unique integer-floored (x, y) cells (x only for HalfCheetah)."""
    dims = 1 if env_name == "half_cheetah" else 2
    return len(np.unique(np.floor(xy.reshape(-1, 2)[:, :dims]), axis=0))
