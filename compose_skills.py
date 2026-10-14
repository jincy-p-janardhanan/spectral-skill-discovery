"""Do the learned skill axes compose? A test for continuous-skill runs (Ant, Quadruped, Humanoid).

For every run: deterministic episodes of the axis skills +-e_k, of every pair (+-e_i +- e_j)/sqrt(2) and of random
unit skills, each twice, with ground-truth outcomes g: the torso's x-y displacement, and for Ant also the net change
of yaw and the mean torso height.

Reported per run:
  new        for each axis k, the share of its effect a_k = (g(+e_k) - g(-e_k)) / 2 that the effects of the axes
             before it cannot explain (1: a new effect, 0: a combination of earlier axes), on standardised outcomes
  n_new      the number of axes with new >= 0.5 and an effect at least 10% of the largest
  purity     for each axis skill, the share of ||phi(s_T) - phi(s_0)||^2 on its own dimension
  additivity outcome of each pair skill against the sum of its axis skills' outcomes, (g(s_i e_i) + g(s_j e_j))
             / sqrt(2), in x-y: relative error and ratio of lengths
  linear R2  share of the variance of held-out random skills' outcomes that a linear map g = W z + b explains

    uv run compose_skills.py --runs "exp/PaperV2/ant_K2_s*" --out figures/compose/ant_K2
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")  # must be set before mujoco is imported

import argparse
import glob
import itertools

import numpy as np
import torch

from make_videos import load_agent
from ssd.envs import PIXEL_ENVS, make_env
from ssd.metrics import quat_to_euler
from ssd.rollout import rollout_chunked


def skills(k: int, n_random: int, rng) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple]]:
    """Axis skills (+e_k, -e_k per k), pair skills with their (axis, sign) parts, and random unit skills."""
    axes = np.concatenate([[e, -e] for e in np.eye(k)])
    pairs, parts = [], []
    for i, j in itertools.combinations(range(k), 2):
        for si, sj in itertools.product((1, -1), repeat=2):
            z = np.zeros(k)
            z[i], z[j] = si / np.sqrt(2), sj / np.sqrt(2)
            pairs.append(z)
            parts.append((2 * i + (si < 0), 2 * j + (sj < 0)))  # rows of the two axis skills in `axes`
    rand = rng.normal(size=(n_random, k))
    rand /= np.linalg.norm(rand, axis=1, keepdims=True)
    return np.concatenate([axes, np.array(pairs).reshape(-1, k), rand]).astype(np.float32), parts, len(axes)


def outcomes(env_name: str, traj: dict) -> tuple[np.ndarray, list[str]]:
    """Ground-truth outcome of each episode: x-y displacement, and for Ant net yaw change and mean height."""
    g = [traj["xy"][:, -1] - traj["xy"][:, 0]]
    names = ["dx", "dy"]
    if env_name == "ant":
        yaw = quat_to_euler(traj["raw"][..., 3:7])[..., 2]
        g += [np.angle(np.exp(1j * np.diff(yaw, axis=1))).sum(axis=1, keepdims=True),
              traj["raw"][..., 2].mean(axis=1, keepdims=True)]
        names += ["dyaw", "height"]
    return np.concatenate(g, axis=1), names


def new_share(effects: np.ndarray) -> np.ndarray:
    """For each row k, the share of ||a_k||^2 outside the span of rows 0..k-1."""
    out = np.ones(len(effects))
    for k in range(1, len(effects)):
        prev = effects[:k].T
        coef, *_ = np.linalg.lstsq(prev, effects[k], rcond=None)
        resid = effects[k] - prev @ coef
        out[k] = resid @ resid / max(effects[k] @ effects[k], 1e-12)
    return out


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(1 - ((y_true - y_pred) ** 2).sum() / ((y_true - y_true.mean(0)) ** 2).sum())


@torch.no_grad()
def evaluate(run_dir: str, n_random: int, repeats: int, device, rng) -> dict:
    agent, run, _ = load_agent(os.path.join(run_dir, "checkpoint.pt"), device)
    env_name, k, horizon = run["env"], run["dim_skill"], run["horizon"]
    z, parts, n_axes = skills(k, n_random, rng)
    pixel = env_name in PIXEL_ENVS
    pool = 8 if pixel else 128
    seed = 555000 + 100 * int(run["seed"])
    envs = [make_env(env_name, seed=seed + i, normalize=bool(run["normalize_obs"])) for i in range(pool)]
    zz = np.repeat(z, repeats, axis=0)
    traj = rollout_chunked(envs, agent.policy, zz, horizon, device, deterministic=True,
                           record_raw=env_name == "ant")
    for env in envs:
        if hasattr(env, "close"):
            env.close()
    ends = torch.as_tensor(traj["obs"][:, [0, -1]], device=device)
    phi = agent.encoder(ends.reshape(-1, ends.shape[-1])).reshape(len(zz), 2, k).cpu().numpy()
    g, names = outcomes(env_name, traj)
    g = g.reshape(len(z), repeats, -1).mean(1)
    dphi = (phi[:, 1] - phi[:, 0]).reshape(len(z), repeats, k).mean(1)

    rand = slice(len(z) - n_random, len(z))
    scale = g[rand].std(0) + 1e-8  # standardise each outcome by its spread over random skills
    gs = g / scale
    effects = (gs[0:n_axes:2] - gs[1:n_axes:2]) / 2  # [K, G]
    new = new_share(effects)
    size = np.linalg.norm(effects, axis=1)
    n_new = int(((new >= 0.5) & (size >= 0.1 * size.max())).sum())
    purity = np.array([dphi[a, a // 2] ** 2 / max((dphi[a] ** 2).sum(), 1e-12) for a in range(n_axes)])
    purity = purity.reshape(k, 2).mean(1)

    xy = g[:, :2]
    pair = xy[n_axes:n_axes + len(parts)]
    pred = np.array([(xy[a] + xy[b]) / np.sqrt(2) for a, b in parts])
    err = np.linalg.norm(pair - pred, axis=1) / np.maximum(np.linalg.norm(pred, axis=1), 1e-8)
    ratio = np.linalg.norm(pair, axis=1) / np.maximum(np.linalg.norm(pred, axis=1), 1e-8)

    zr, gr = z[rand], gs[rand]
    n_fit = int(0.75 * n_random)
    a = np.c_[zr, np.ones(n_random)]
    w, *_ = np.linalg.lstsq(a[:n_fit], gr[:n_fit], rcond=None)
    lin_r2 = r2(gr[n_fit:], a[n_fit:] @ w)
    lin_r2_xy = r2(gr[n_fit:, :2], (a[n_fit:] @ w)[:, :2])
    return dict(run=os.path.basename(run_dir), env=env_name, k=k, names=names, z=z, g=g, dphi=dphi,
                effects=effects, new=new, size=size, n_new=n_new, purity=purity, pair_err=err, pair_ratio=ratio,
                lin_r2=lin_r2, lin_r2_xy=lin_r2_xy, axis_xy=xy[:n_axes])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", required=True, help="glob over run directories with checkpoint.pt")
    p.add_argument("--n_random", type=int, default=200)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--out", required=True, help="output prefix: <out>.txt and <out>_<run>.npz")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    runs = sorted((d for d in glob.glob(args.runs) if os.path.exists(os.path.join(d, "checkpoint.pt"))),
                  key=lambda d: int(d.rsplit("_s", 1)[1]))
    lines = [f"{args.runs}: {args.n_random} random skills, {args.repeats} episodes per skill",
             "run | n_new | new share per axis | |effect| per axis | purity per axis | pair err (median) | "
             "pair length ratio (median, IQR) | linear R2 (all, x-y)"]
    for d in runs:
        r = evaluate(d, args.n_random, args.repeats, device, rng)
        np.savez(f"{args.out}_{r['run']}.npz", **{k: v for k, v in r.items() if k not in ("run", "env", "names")},
                 names=np.array(r["names"]))
        q1, q2, q3 = np.percentile(r["pair_ratio"], [25, 50, 75])
        lines.append(f"{r['run']:13s} {r['n_new']} | {' '.join(f'{x:.2f}' for x in r['new'])} | "
                     f"{' '.join(f'{x:.2f}' for x in r['size'])} | {' '.join(f'{x:.2f}' for x in r['purity'])} | "
                     f"{np.median(r['pair_err']):.2f} | {q2:.2f} ({q1:.2f}-{q3:.2f}) | "
                     f"{r['lin_r2']:.2f} {r['lin_r2_xy']:.2f}")
        print(lines[-1], flush=True)
    with open(f"{args.out}.txt", "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
