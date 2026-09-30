import numpy as np
import torch


@torch.no_grad()
def rollout(envs, policy, z: np.ndarray, horizon: int, device, deterministic: bool = False,
            record_raw: bool = False, render: bool = False) -> dict[str, np.ndarray]:
    """One episode per env in lockstep, env i with skill z[i].

    record_raw adds the unnormalised states as "raw"; render adds "frames" (needs render_mode="rgb_array").
    """
    n = len(envs)
    obs = np.stack([env.reset()[0] for env in envs])
    xy0 = np.stack([env.unwrapped._xy() for env in envs])
    raw = [[env.unwrapped._raw_obs()] for env in envs] if record_raw else None
    frames = [[env.render()] for env in envs] if render else None
    z_t = torch.as_tensor(z, dtype=torch.float32, device=device)

    obs_dtype = np.uint8 if envs[0].observation_space.dtype == np.uint8 else np.float32  # images stay uint8
    obs_buf = np.empty((n, horizon + 1, obs.shape[-1]), dtype=obs_dtype)
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
            if record_raw:
                raw[i].append(env.unwrapped._raw_obs())
            if render:
                frames[i].append(env.render())
    out = {"obs": obs_buf, "act": act_buf, "done": done_buf, "z": np.asarray(z, dtype=np.float32), "xy": xy_buf}
    if record_raw:
        out["raw"] = np.asarray(raw, dtype=np.float32)
    if render:
        out["frames"] = np.asarray(frames, dtype=np.uint8)
    return out


SWEEP_ALPHAS = (-2.0, -1.0, 0.0, 1.0, 2.0)


@torch.no_grad()
def dimension_sweep(envs, policy, encoder, dim_skill: int, horizon: int, device,
                    alphas=SWEEP_ALPHAS) -> dict[str, np.ndarray]:
    """Deterministic rollouts of z = alpha * e_k for each dimension k, cycling through alphas over the envs.

    Returns xy [K, n, T+1, 2], dphi [K, n, T+1] = phi_k(s_t) - phi_k(s_0), and alpha [n].
    """
    n = len(envs)
    alpha = np.resize(np.asarray(alphas, dtype=np.float32), n)
    xy, dphi = [], []
    for k in range(dim_skill):
        z = np.zeros((n, dim_skill), dtype=np.float32)
        z[:, k] = alpha
        traj = rollout(envs, policy, z, horizon, device, deterministic=True)
        obs = torch.as_tensor(traj["obs"], device=device)
        phi_k = encoder(obs.reshape(-1, obs.shape[-1]))[:, k].reshape(n, horizon + 1)
        dphi.append((phi_k - phi_k[:, :1]).cpu().numpy())
        xy.append(traj["xy"])
    return {"xy": np.stack(xy), "dphi": np.stack(dphi), "alpha": alpha}


def rollout_chunked(envs, policy, z: np.ndarray, horizon: int, device, **kwargs) -> dict[str, np.ndarray]:
    """rollout() for len(z) episodes on a smaller pool of envs. Adds "success" for envs that track it."""
    parts, success = [], []
    for start in range(0, len(z), len(envs)):
        chunk = z[start:start + len(envs)]
        parts.append(rollout(envs[:len(chunk)], policy, chunk, horizon, device, **kwargs))
        if hasattr(envs[0], "success"):
            success += [env.success.copy() for env in envs[:len(chunk)]]
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    if success:
        out["success"] = np.array(success)
    return out
