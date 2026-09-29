import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt

from ssd.spectral import Spectrum


def skill_colors(z: np.ndarray) -> np.ndarray:
    """Hue from the angle of the two leading skill dims, saturation from their norm.
    One-hot (discrete) skills get one distinct colour per skill instead."""
    if np.all((z == 0) | (z == 1)) and np.all(z.sum(axis=1) == 1):
        idx, k = z.argmax(axis=1), z.shape[1]
        if k <= 20:
            return plt.get_cmap("tab20" if k > 10 else "tab10")(idx)[:, :3]
        return mcolors.hsv_to_rgb(np.c_[idx / k, np.full(len(z), 0.75), np.full(len(z), 0.85)])
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


ALPHA_COLORS = {-2.0: "#184f95", -1.0: "#6da7ec", 0.0: "#9b9a96", 1.0: "#ef8e8d", 2.0: "#b8292a"}


def plot_dimension_sweeps(sweep: dict, env_name: str, axis_lim: float | None = None, title: str | None = None):
    """One column per dimension k for z = alpha * e_k: x-y paths (top), phi_k(s_t) - phi_k(s_0) (bottom)."""
    xy, dphi, alpha = sweep["xy"], sweep["dphi"], sweep["alpha"]
    k_dims, n, t1 = dphi.shape
    fig, axes = plt.subplots(2, k_dims, figsize=(2.9 * k_dims, 6.2), squeeze=False,
                             gridspec_kw={"height_ratios": [1.15, 1]})
    lim = axis_lim or max(1.0, np.abs(xy).max() * 1.05)
    ylim = np.abs(dphi).max() * 1.05 or 1.0
    for k in range(k_dims):
        ax = axes[0, k]
        for i in range(n):
            y = xy[k, i, :, 1] if env_name != "half_cheetah" else np.full(t1, (i - n / 2) / 1.25)
            ax.plot(xy[k, i, :, 0], y, color=ALPHA_COLORS.get(float(alpha[i]), "#52514e"), lw=1.2)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_title(f"dimension {k + 1}", fontsize=10, fontweight="bold", loc="left")
        ax.tick_params(labelsize=7)
        ax = axes[1, k]
        for i in range(n):
            ax.plot(np.arange(t1), dphi[k, i], color=ALPHA_COLORS.get(float(alpha[i]), "#52514e"), lw=1.2)
        ax.set_ylim(-ylim, ylim)
        ax.axhline(0, color="#c3c2b7", lw=0.8)
        ax.set_xlabel("step t", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0, 0].set_ylabel("y (m)", fontsize=8)
    axes[1, 0].set_ylabel(r"$\phi_k(s_t)-\phi_k(s_0)$", fontsize=8)
    handles = [plt.Line2D([], [], color=c, lw=2, label=f"α = {a:+g}") for a, c in ALPHA_COLORS.items()]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(title or "Skills z = α·eₖ, one dimension at a time", x=0.01, y=0.955, ha="left",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
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
