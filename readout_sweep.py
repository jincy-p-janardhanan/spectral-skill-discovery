"""Re-run the spectrum readout offline on finished runs to check how stable K* is.

    uv run readout_sweep.py --runs "exp/PaperV2/half_cheetah_*" --out figures/readout/half_cheetah.txt

For each number of trajectories N and quantile, repeats the readout on random subsets of each run's replay
buffer and reports the K* frequencies and how often each seed's repeats agree.
"""

import argparse
import glob
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from ssd import spectral
from ssd.envs import PIXEL_ENVS
from ssd.networks import Encoder


def load_run(run_dir: str):
    """Encode a run's replay buffer: phi [n, T+1, K], skill weights [n, K] and the iteration."""
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    state = torch.load(os.path.join(run_dir, "resume.pt"), map_location="cpu", weights_only=False)
    obs, z = state["buffer"]["obs"], state["buffer"]["z"]  # pixels: newest frame only, stacks rebuilt below
    k = z.shape[1]
    n, t1 = obs.shape[:2]
    pixel = cfg["env"] in PIXEL_ENVS
    obs_dim = state["agent"]["encoder"]["input_idx"].numel() if pixel else obs.shape[-1]
    stack = obs_dim // obs.shape[-1]
    pixel_shape = (obs_dim // (64 * 64), 64, 64) if pixel else None  # stacked 64x64 RGB frames, kept uint8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = Encoder(obs_dim, k, cfg["hidden_dim"], cfg["n_layers"], pixel_shape=pixel_shape,
                  per_dim=bool(cfg.get("encoder_per_dim", 0))).to(device)
    enc.load_state_dict(state["agent"]["encoder"])
    steps = (torch.arange(t1)[:, None] + torch.arange(1 - stack, 1)).clamp(min=0)  # as ReplayBuffer._stacked
    phi = []
    with torch.no_grad():
        for i in range(0, n, 64):
            flat = obs[i:i + 64][:, steps].flatten(-2).reshape(-1, obs_dim)
            chunks = flat.split(2048 if pixel else 65536)
            phi.append(torch.cat([enc(c.to(device) if pixel else c.float().to(device)).cpu() for c in chunks]))
    phi = torch.cat(phi).reshape(n, t1, k)
    if cfg.get("discrete") and not (cfg.get("cube_skills") or cfg.get("signed_skills")):
        z = (z - z.mean(dim=-1, keepdim=True)) * k / max(k - 1, 1)  # METRA's centred mask, as in training
    return phi, z, state["iter"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", required=True, help="glob over run directories with resume.pt")
    p.add_argument("--n_trajs", type=int, nargs="+", default=[128, 256, 512, 1024, 2048])
    p.add_argument("--quantiles", type=float, nargs="+", default=[0.95, 0.99])
    p.add_argument("--n_shuffles", type=int, default=1000)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--min_ratio", type=float, default=0.0, help="effect-size condition R_k >= r * max R")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    runs = [d for d in sorted(glob.glob(args.runs)) if os.path.exists(os.path.join(d, "resume.pt"))]
    data = []
    for d in runs:
        phi, z, it = load_run(d)
        data.append((os.path.basename(d), phi, z))
        print(f"loaded {d}: {phi.shape[0]} trajectories, iteration {it}", flush=True)

    lines = [f"{len(runs)} runs ({args.runs}), M = {args.n_shuffles}, {args.repeats} repeats per setting, "
             f"effect-size ratio {args.min_ratio}",
             f"{'N':>5} {'q':>5} | K* frequency over seeds x repeats | per-seed modal K* | agreement with mode"]
    for n in args.n_trajs:
        for q in args.quantiles:
            all_k, modes, agree = Counter(), [], []
            for name, phi, z in data:
                rng = np.random.default_rng(0)
                ks = []
                for r in range(args.repeats):
                    idx = torch.as_tensor(rng.choice(phi.shape[0], size=min(n, phi.shape[0]), replace=False))
                    spec = spectral.spectrum_readout(phi[idx], z[idx], args.n_shuffles, q,
                                                     torch.Generator().manual_seed(r))
                    passed = spec.returns > spec.floor
                    if args.min_ratio > 0:
                        passed &= spec.returns >= args.min_ratio * spec.returns.max()
                    ks.append(int(passed.int().cumprod(dim=0).sum()))
                c = Counter(ks)
                mode, count = c.most_common(1)[0]
                all_k.update(ks)
                modes.append(mode)
                agree.append(count / len(ks))
            freq = " ".join(f"{k}:{v / sum(all_k.values()):.2f}" for k, v in sorted(all_k.items()))
            lines.append(f"{n:5d} {q:5.2f} | {freq:34s} | {' '.join(map(str, modes)):17s} | {np.mean(agree):.2f}")
            print(lines[-1], flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
