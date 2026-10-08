"""Videos of the learned skills: SSD's final policies next to METRA's final evaluation videos.

The SSD skills follow METRA's evaluation videos (Video_RandomZ): 2-D skills use the 8 compass directions and z = 0,
laid out as a compass; other continuous skills use -e_k and +e_k, one row per dimension; discrete skills use every
skill. Every skill is rolled out twice (once for Kitchen's 32 skills) with the deterministic policy. Each tile shows
z and the displacement phi_k(s_t) - phi_k(s_0) of every dimension k (Kitchen: the tasks achieved so far), and a map
of the paths (HalfCheetah: x over time) is drawn next to the tiles in the same colours as the tile borders.

Per environment the output directory holds ssd_s<seed>.mp4 for every seed with its last frame as a PNG, METRA's
final evaluation video metra_s<seed>.mp4, and side_by_side.mp4 with the median seed of each method (by final
coverage), time-aligned.

    uv run make_videos.py --ssd_dir exp/PaperV2 --metra_dir <metra_run_group> --envs ant half_cheetah \
        --out figures/videos
    uv run make_videos.py --ssd_dir exp/PaperV3 --metra_dir <metra_run_group> \
        --envs dmc_quadruped dmc_humanoid kitchen --out figures/videos
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")  # must be set before mujoco is imported

import argparse
import dataclasses
import glob
import re
import shutil
from pathlib import Path

import imageio.v3 as iio
import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch
from matplotlib import font_manager
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from compare_metra import ENV_TITLES, metra_runs, ssd_runs
from ssd.algorithm import SpectralSkillDiscovery, SSDConfig
from ssd.envs import PIXEL_ENVS, encoder_input_idx, make_env
from ssd.plotting import skill_colors

FONT = font_manager.findfont("DejaVu Sans")
TASK_LETTERS = "BLSHMK"  # BottomBurner, LightSwitch, SlideCabinet, HingeCabinet, Microwave, Kettle
FPS = 15  # as METRA's videos


def load_agent(path: str, device) -> tuple[SpectralSkillDiscovery, dict, dict]:
    """The encoder and policy of a checkpoint, with the run's config."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    run = ckpt["config"]
    cfg = SSDConfig(**{f.name: f.type(run[f.name]) for f in dataclasses.fields(SSDConfig) if run.get(f.name) is not None})
    env = make_env(run["env"], seed=0, normalize=bool(run["normalize_obs"]))
    obs_dim, space = env.observation_space.shape[0], env.action_space
    agent = SpectralSkillDiscovery(cfg, obs_dim, space.shape[0], space.low, space.high, device,
                                   encoder_input_idx=encoder_input_idx(run["env"], obs_dim,
                                                                       bool(run.get("encoder_hide_pos", 0))),
                                   pixel_shape=getattr(env, "pixel_shape", None))
    agent.encoder.load_state_dict(ckpt["agent"]["encoder"])
    agent.sac.policy.load_state_dict(ckpt["agent"]["sac"]["policy"])
    agent.encoder.eval()
    return agent, run, ckpt


def video_skills(run: dict, k: int) -> tuple[np.ndarray, list[str], int]:
    """The skills of the video, their labels and the number of grid columns."""
    if run["discrete"] and run.get("cube_skills"):
        idx = np.arange(2 ** k)
        signs = 2 * ((idx[:, None] >> np.arange(k)) & 1) - 1
        z, labels = signs / np.sqrt(k), ["z: " + " ".join("+" if s > 0 else "-" for s in row) for row in signs]
        repeats, cols = (2, 8) if len(z) <= 16 else (1, 8)
    elif run["discrete"]:
        z, labels = np.eye(k), [f"z = e{i + 1}" for i in range(k)]
        repeats, cols = 2, 8
    elif k == 2:  # METRA's compass: 135, 90, 45 / 180, z = 0, 0 / 225, 270, 315 degrees
        angles = [3, 2, 1, 4, None, 0, 5, 6, 7]
        z = np.array([[0.0, 0.0] if a is None else [np.cos(a * np.pi / 4), np.sin(a * np.pi / 4)] for a in angles])
        labels = ["z = 0 (not trained)" if a is None else f"z: {45 * a} deg" for a in angles]
        repeats, cols = 2, 6
    else:
        z = np.concatenate([[-e, e] for e in np.eye(k)])
        labels = [f"z = {s}e{i + 1}" for i in range(k) for s in "-+"]
        repeats, cols = 2, 4
    return np.repeat(z, repeats, axis=0).astype(np.float32), [l for l in labels for _ in range(repeats)], cols


@torch.no_grad()
def record(env_name: str, agent, z: np.ndarray, horizon: int, device, size: int, pool: int, seed: int,
           normalize: bool) -> dict:
    """Deterministic episodes with skill z[i] on a pool of envs: frames, x-y, phi and the tasks achieved."""
    pixel = env_name in PIXEL_ENVS
    kwargs = {} if pixel else dict(render_mode="rgb_array", width=size, height=size, camera_name="track")
    envs = [make_env(env_name, seed=seed + i, normalize=normalize, **kwargs) for i in range(min(pool, len(z)))]
    as_obs = (lambda o: o) if pixel else (lambda o: o.astype(np.float32))  # images stay uint8
    frames, xy, obs, success = [], [], [], []
    for start in range(0, len(z), len(envs)):
        zc = torch.as_tensor(z[start:start + len(envs)], device=device)
        chunk = envs[:len(zc)]
        o = as_obs(np.stack([env.reset()[0] for env in chunk]))
        fr = [[env.render(size) if pixel else env.render()] for env in chunk]
        pos = [[env.unwrapped._xy()] for env in chunk]
        ob, su = [o], [[getattr(env, "success", np.zeros(0)).copy() for env in chunk]]
        for _ in range(horizon):
            act = agent.policy(torch.as_tensor(o, device=device), zc, deterministic=True)[0].cpu().numpy()
            steps = [env.step(a) for env, a in zip(chunk, act)]
            o = as_obs(np.stack([s[0] for s in steps]))
            for i, env in enumerate(chunk):
                fr[i].append(env.render(size) if pixel else env.render())
                pos[i].append(steps[i][4]["next_coordinates"])
            ob.append(o)
            su.append([getattr(env, "success", np.zeros(0)).copy() for env in chunk])
        frames += fr
        xy += pos
        obs.append(np.stack(ob, axis=1))
        success.append(np.stack([np.stack(s) for s in su], axis=1))
    obs = np.concatenate(obs)  # [n, T+1, obs_dim]
    flat = torch.as_tensor(obs.reshape(-1, obs.shape[-1]), device=device)
    phi = torch.cat([agent.encoder(c) for c in flat.split(2048)]).reshape(len(z), horizon + 1, -1).cpu().numpy()
    for env in envs:
        if hasattr(env, "close"):
            env.close()
    return {"frames": np.asarray(frames, dtype=np.uint8), "xy": np.asarray(xy, dtype=np.float32),
            "dphi": phi - phi[:, :1], "success": np.concatenate(success)}


def tile(frame: np.ndarray, size: int, color, lines: list[tuple[str, tuple]], font) -> np.ndarray:
    """A frame resized to size x size with a coloured border and text lines (text, fill) at the top."""
    img = Image.fromarray(frame).resize((size, size), Image.BILINEAR)
    draw = ImageDraw.Draw(img)
    y = 3
    for text, fill in lines:
        draw.text((5, y), text, font=font, fill=fill, stroke_width=2, stroke_fill=(255, 255, 255))
        y += font.size + 3
    out = Image.new("RGB", (size + 6, size + 6), tuple(int(255 * c) for c in color))
    out.paste(img, (3, 3))
    return np.asarray(out)


def fmt(v: float) -> str:
    return f"{v:+.1f}" if abs(v) < 10 else f"{v:+.0f}"


class PathMap:
    """Matplotlib map of the paths up to time t: x-y, or x over time for HalfCheetah."""

    def __init__(self, env: str, xy: np.ndarray, colors: np.ndarray, height: int):
        self.env, self.xy = env, xy
        self.fig = plt.figure(figsize=(height / 100, height / 100), dpi=100)
        ax = self.fig.add_subplot()
        t = np.arange(xy.shape[1])
        self.lines = [ax.plot([], [], color=c, lw=1.2)[0] for c in colors]
        self.dots = [ax.plot([], [], "o", color=c, ms=3)[0] for c in colors]
        if env == "half_cheetah":
            lo, hi = xy[..., 0].min(), xy[..., 0].max()
            ax.set_xlim(lo - 0.05 * (hi - lo + 1), hi + 0.05 * (hi - lo + 1))
            ax.set_ylim(t[-1], 0)
            ax.set_xlabel("x")
            ax.set_ylabel("time step")
        else:
            r = np.abs(xy).max() * 1.08 + 1e-3
            ax.set_xlim(-r, r)
            ax.set_ylim(-r, r)
            ax.set_aspect("equal")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
        ax.grid(alpha=0.3)
        self.fig.tight_layout()
        self.height = height

    def frame(self, t: int) -> np.ndarray:
        for i, (line, dot) in enumerate(zip(self.lines, self.dots)):
            x = self.xy[i, :t + 1, 0]
            y = np.arange(t + 1) if self.env == "half_cheetah" else self.xy[i, :t + 1, 1]
            line.set_data(x, y)
            dot.set_data(x[-1:], y[-1:])
        self.fig.canvas.draw()
        img = np.asarray(self.fig.canvas.buffer_rgba())[..., :3]
        return np.asarray(Image.fromarray(img).resize((self.height, self.height), Image.BILINEAR))


def header(width: int, text: str, font) -> np.ndarray:
    img = Image.new("RGB", (width, font.size + 14), (245, 245, 245))
    ImageDraw.Draw(img).text((8, 6), text, font=font, fill=(20, 20, 20))
    return np.asarray(img)


def pad16(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    return np.pad(frame, ((0, -h % 16), (0, -w % 16), (0, 0)), constant_values=255)


def write_video(path: Path, frames) -> np.ndarray:
    """Streams frames (an iterable of equal-size RGB arrays) to an H.264 MP4 and returns the last frame."""
    writer = f = None
    for f in frames:
        f = pad16(f)
        if writer is None:
            writer = imageio_ffmpeg.write_frames(str(path), (f.shape[1], f.shape[0]), fps=FPS, codec="libx264",
                                                 quality=7, macro_block_size=1)
            writer.send(None)
        writer.send(np.ascontiguousarray(f))
    writer.close()
    return f


def ssd_frames(env: str, rec: dict, z: np.ndarray, labels: list[str], cols: int, title: str, size: int):
    """Generator of the SSD video frames: header, tile grid and path map."""
    font = ImageFont.truetype(FONT, 12)
    big = ImageFont.truetype(FONT, 16)
    colors = skill_colors(z)
    n, t1 = rec["frames"].shape[:2]
    rows = -(-n // cols)
    pathmap = None if env == "kitchen" else PathMap(env, rec["xy"], colors, rows * (size + 6))
    for t in range(t1):
        tiles = []
        for i in range(n):
            d = [fmt(v) for v in rec["dphi"][i, t]]  # three dimensions per line
            lines = [(labels[i], (20, 20, 20))] + [(("Δφ " if j == 0 else "     ") + " ".join(d[j:j + 3]), (20, 20, 120))
                                                   for j in range(0, len(d), 3)]
            if env == "kitchen":
                done = "".join(c if s else "." for c, s in zip(TASK_LETTERS, rec["success"][i, t]))
                lines.append(("tasks " + done, (0, 120, 0)))
            tiles.append(tile(rec["frames"][i, t], size, colors[i], lines, font))
        blank = np.full_like(tiles[0], 255)
        tiles += [blank] * (rows * cols - n)
        grid = np.concatenate([np.concatenate(tiles[r * cols:(r + 1) * cols], axis=1) for r in range(rows)])
        if pathmap is not None:
            grid = np.concatenate([grid, pathmap.frame(t)], axis=1)
        yield np.concatenate([header(grid.shape[1], f"{title}   t = {t}", big), grid])
    if pathmap is not None:
        plt.close(pathmap.fig)


def seed_of(path: str) -> int:
    return int(re.findall(r"_s(\d+)$", path.rstrip("/"))[0])


def last_metra_video(run_dir: str) -> str | None:
    """The final Video_RandomZ of a run, following resumed_from.txt back if the run saved none."""
    while run_dir:
        videos = glob.glob(os.path.join(run_dir, "plots", "Video_RandomZ_*.mp4"))
        if videos:
            return max(videos, key=lambda q: int(re.findall(r"_(\d+)\.mp4$", q)[0]))
        rf = os.path.join(run_dir, "resumed_from.txt")
        run_dir = os.path.realpath(open(rf).read().split()[0]) if os.path.exists(rf) else None
    return None


def side_by_side(ssd_path: Path, metra_path: str, ssd_title: str, metra_title: str):
    """Generator of frames with SSD's video next to METRA's, scaled to the same height and time-aligned."""
    a, b = iio.imread(ssd_path), iio.imread(metra_path)
    font = ImageFont.truetype(FONT, 16)
    h = a.shape[1]
    scale = h / b.shape[1]
    b_size = (int(round(b.shape[2] * scale)), h)
    n = max(len(a), len(b))
    for i in range(n):
        fa = a[min(len(a) - 1, i * len(a) // n)]
        fb = np.asarray(Image.fromarray(b[min(len(b) - 1, i * len(b) // n)]).resize(b_size, Image.NEAREST))
        left = np.concatenate([header(fa.shape[1], ssd_title, font), fa])
        right = np.concatenate([header(fb.shape[1], metra_title, font), fb])
        gap = np.full((left.shape[0], 12, 3), 255, dtype=np.uint8)
        yield np.concatenate([left, gap, right], axis=1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ssd_dir", required=True, help="SSD run group, e.g. exp/PaperV2")
    p.add_argument("--metra_dir", default=None, help="METRA experiment group with the baseline runs")
    p.add_argument("--envs", nargs="+", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    p.add_argument("--out", default="figures/videos")
    p.add_argument("--size", type=int, default=180, help="tile size in pixels")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    for env in args.envs:
        out = Path(args.out) / env
        out.mkdir(parents=True, exist_ok=True)
        size = args.size
        metric = "tasks achieved" if env == "kitchen" else "coverage"
        runs = [r for r in ssd_runs(args.ssd_dir, env) if seed_of(r["dir"]) in args.seeds]
        final = {seed_of(r["dir"]): r["value"][-1] for r in runs}
        med_ssd = seed_of(runs[int(np.argsort([r["value"][-1] for r in runs])[len(runs) // 2])]["dir"])
        index = [f"{ENV_TITLES.get(env, env)}: SSD runs in {args.ssd_dir}"]
        for r in runs:
            seed = seed_of(r["dir"])
            agent, run, ckpt = load_agent(os.path.join(r["dir"], "checkpoint.pt"), args.device)
            z, labels, cols = video_skills(run, agent.cfg.dim_skill)
            pool = 4 if env == "kitchen" else (8 if env in PIXEL_ENVS else len(z))
            rec = record(env, agent, z, run["horizon"], args.device, 240 if env in PIXEL_ENVS else size,
                         pool, seed=777_000 + 100 * seed, normalize=bool(run["normalize_obs"]))
            title = (f"SSD  {ENV_TITLES.get(env, env)}  seed {seed}  {ckpt['env_steps'] / 1e6:.1f}M steps  "
                     f"final {metric} {final[seed]:g}{'  (median seed)' if seed == med_ssd else ''}")
            path = out / f"ssd_s{seed}.mp4"
            last = write_video(path, ssd_frames(env, rec, z, labels, cols, title, size))
            Image.fromarray(last).save(out / f"ssd_s{seed}_last.png")
            endpoint = np.abs(rec["dphi"][:, -1]).mean(axis=0)
            index.append(f"  seed {seed}: {metric} {final[seed]:g}, mean |dphi_k(T)| over the video skills "
                         + " ".join(f"{v:.1f}" for v in endpoint) + ("  <- median" if seed == med_ssd else ""))
            print(f"Wrote {path}", flush=True)

        if args.metra_dir:
            metra = [c for c in metra_runs(args.metra_dir, env).get("METRA", []) if c["seed"] in args.seeds]
            index.append(f"METRA runs in {args.metra_dir}")
            if metra:
                med = metra[int(np.argsort([c["value"][-1] for c in metra])[len(metra) // 2])]
                for c in metra:
                    video = last_metra_video(c["dir"])
                    index.append(f"  seed {c['seed']}: {metric} {c['value'][-1]:g}, video {video}"
                                 + ("  <- median" if c is med else ""))
                    if video:
                        shutil.copy(video, out / f"metra_s{c['seed']}.mp4")
                video = last_metra_video(med["dir"])
                if video:
                    write_video(out / "side_by_side.mp4", side_by_side(
                        out / f"ssd_s{med_ssd}.mp4", video,
                        f"SSD, median seed {med_ssd} ({metric} {final[med_ssd]:g})",
                        f"METRA, median seed {med['seed']} ({metric} {med['value'][-1]:g}), its own evaluation video"))
                    print(f"Wrote {out / 'side_by_side.mp4'}", flush=True)
        (out / "index.txt").write_text("\n".join(index) + "\n")
        print("\n".join(index), flush=True)


if __name__ == "__main__":
    main()
