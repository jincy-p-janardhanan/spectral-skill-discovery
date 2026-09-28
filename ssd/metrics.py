# Evaluation metrics on recorded trajectories.

import numpy as np


def quat_to_euler(q: np.ndarray) -> np.ndarray:
    """MuJoCo quaternion (w, x, y, z) -> (roll, pitch, yaw) in radians, [..., 3]."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.stack([roll, pitch, yaw], axis=-1)


def coverage(xy: np.ndarray, env_name: str) -> int:
    """Number of unique integer (x, y) cells visited (x only for HalfCheetah)."""
    dims = 1 if env_name == "half_cheetah" else 2
    return len(np.unique(np.floor(xy.reshape(-1, 2)[:, :dims]), axis=0))


def body_coverage(raw: np.ndarray, env, angle_bin: float = np.pi / 6, joint_bins: int = 5) -> dict[str, int]:
    """Unique cells of torso orientation and of limb joint angles over raw observations [N, T+1, obs_dim]."""
    flat = raw.reshape(-1, raw.shape[-1])
    orient = np.floor(env.orientation(flat) / angle_bin)
    lo, hi = env.joint_range.T
    joints = np.clip(np.floor((env.joint_angles(flat) - lo) / (hi - lo) * joint_bins), 0, joint_bins - 1)
    return {"orientation_coverage": len(np.unique(orient, axis=0)),
            "posture_coverage": len(np.unique(joints, axis=0))}
