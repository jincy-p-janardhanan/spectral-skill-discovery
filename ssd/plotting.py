import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt

from ssd.spectral import Spectrum


def skill_colors(z: np.ndarray) -> np.ndarray:
    """Hue from the angle of the two leading skill dims, saturation from their norm."""
    z2 = z[:, :2] if z.shape[1] >= 2 else np.c_[z[:, 0], np.zeros(len(z))]
    hue = (np.arctan2(z2[:, 1], z2[:, 0]) / (2 * np.pi)) % 1.0
    sat = np.clip(np.linalg.norm(z2, axis=1) / 2.0, 0.25, 1.0)
    return mcolors.hsv_to_rgb(np.c_[hue, sat, np.full(len(z), 0.85)])


def plot_trajectories(xy: np.ndarray, z: np.ndarray, env_name: str, axis_lim: float | None = None):
    fig, ax = plt.subplots(figsize=(5, 5))
    cols = skill_colors(z)
    for i in range(len(xy)):
        if env_name == "half_cheetah":  # 1-D, spread vertically
            ax.plot(xy[i, :, 0], np.full(xy.shape[1], (i - len(xy) / 2) / 1.25), color=cols[i], lw=0.8)
        else:
            ax.plot(xy[i, :, 0], xy[i, :, 1], color=cols[i], lw=0.8)
    if axis_lim is not None:
        ax.set_xlim(-axis_lim, axis_lim)
        ax.set_ylim(-axis_lim, axis_lim)
    ax.set_aspect("equal")
    ax.set_title("x-y trajectories (color: z1, z2)")
    fig.tight_layout()
    return fig


def plot_spectrum(spec: Spectrum):
    sigma2, ret, floor = (x.cpu().numpy() for x in (spec.sigma2, spec.returns, spec.floor))
    k = np.arange(1, len(sigma2) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
    ax1.bar(k, sigma2, color="#4c72b0")
    ax1.set_title(r"spread $\sigma_k^2$")
    ax1.set_xlabel("dimension k")
    ax2.bar(k, ret, color=["#4c72b0" if i < spec.k_star else "#bbbbbb" for i in range(len(k))], label="$R_k$")
    ax2.plot(k, floor, "k_", ms=18, mew=2, label="95% permutation floor")
    ax2.axvline(spec.k_star + 0.5, color="#c44e52", ls="--", lw=1)
    ax2.set_title(rf"per-dimension return $R_k$  ($K^\star$ = {spec.k_star})")
    ax2.set_xlabel("dimension k")
    ax2.legend(fontsize=8)
    for ax in (ax1, ax2):
        ax.set_xticks(k)
    fig.tight_layout()
    return fig
