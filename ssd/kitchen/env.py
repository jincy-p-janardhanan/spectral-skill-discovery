"""Pixel-based Franka Kitchen with METRA's setup: end-effector control, 64x64 renders, 3 stacked frames."""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import threading
from collections import deque
from itertools import combinations

import numpy as np
from gymnasium import spaces

GOAL_NAMES = ["BottomBurner", "LightSwitch", "SlideCabinet", "HingeCabinet", "Microwave", "Kettle"]


def get_kitchen_benchmark_goals():
    """LEXA's kitchen goals: 6 single-object, then 6 two-object."""
    object_goal_vals = {"bottom_burner": [-0.88, -0.01], "light_switch": [-0.69, -0.05], "slide_cabinet": [0.37],
                        "hinge_cabinet": [0., 0.5], "microwave": [-0.5], "kettle": [-0.23, 0.75, 1.62]}
    object_goal_idxs = {"bottom_burner": [9, 10], "light_switch": [17, 18], "slide_cabinet": [19],
                        "hinge_cabinet": [20, 21], "microwave": [22], "kettle": [23, 24, 25]}
    base = ["bottom_burner", "light_switch", "slide_cabinet", "hinge_cabinet", "microwave", "kettle"]
    configs = [[base[i]] for i in range(6)] + [[base[i], base[j]] for i, j in combinations([1, 2, 3, 5], 2)]
    goals = [np.concatenate([object_goal_vals[o] for o in objs]) for objs in configs]
    idxs = [np.concatenate([object_goal_idxs[o] for o in objs]) for objs in configs]
    return goals, idxs, configs


class KitchenPixelEnv:
    size = 64
    frame_stack = 3
    action_repeat = 1
    pixel_shape = (frame_stack * 3, size, size)
    LOCK = threading.Lock()

    def __init__(self, seed: int = 0):
        from ssd.kitchen.kitchen_envs import KitchenMicrowaveKettleLightTopLeftBurnerV0

        with self.LOCK:
            self._env = KitchenMicrowaveKettleLightTopLeftBurnerV0(
                frame_skip=40, control_mode="end_effector", imwidth=self.size, imheight=self.size)
            self._env.sim_robot.renderer._camera_settings = dict(
                distance=1.86, lookat=[-0.3, .5, 2.], azimuth=90, elevation=-60)
        self._env.reset()
        self.init_qpos = self._env.sim.data.qpos.copy()
        self.goals, self.goal_idxs, self.goal_configs = get_kitchen_benchmark_goals()
        self.action_space = spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Box(0, 255, shape=(int(np.prod(self.pixel_shape)),), dtype=np.uint8)
        self._frames: deque = deque(maxlen=self.frame_stack)
        self.success = np.zeros(len(GOAL_NAMES))

    @property
    def unwrapped(self):
        return self

    def compute_success(self, goal: int) -> int:
        """LEXA's task-relevant success of goal `goal` (object thresholds on the current qpos)."""
        qpos = self._env.sim.data.qpos.copy()
        goal_qpos = self.init_qpos.copy()
        goal_qpos[self.goal_idxs[goal]] = self.goals[goal]
        per_obj = {
            "bottom_burner": ((qpos[9] < -0.38) and (goal_qpos[9] < -0.38)) or ((qpos[9] > -0.38) and (goal_qpos[9] > -0.38)),
            "top_burner": ((qpos[13] < -0.38) and (goal_qpos[13] < -0.38)) or ((qpos[13] > -0.38) and (goal_qpos[13] > -0.38)),
            "light_switch": ((qpos[17] < -0.25) and (goal_qpos[17] < -0.25)) or ((qpos[17] > -0.25) and (goal_qpos[17] > -0.25)),
            "slide_cabinet": abs(qpos[19] - goal_qpos[19]) < 0.1,
            "hinge_cabinet": abs(qpos[21] - goal_qpos[21]) < 0.2,
            "microwave": abs(qpos[22] - goal_qpos[22]) < 0.2,
            "kettle": np.linalg.norm(qpos[23:25] - goal_qpos[23:25]) < 0.2,
        }
        return int(all(per_obj[o] for o in self.goal_configs[goal]))

    def _render(self) -> np.ndarray:
        img = self._env.render("rgb_array", width=self.size, height=self.size)
        return np.ascontiguousarray(img.transpose(2, 0, 1))

    def _obs(self) -> np.ndarray:
        return np.concatenate(list(self._frames), axis=0).reshape(-1)

    def _xy(self) -> np.ndarray:
        return self._env.sim.data.qpos[:2].copy()  # METRA plots the first two robot joints

    def _update_success(self):
        self.success = np.maximum(self.success, [self.compute_success(g) for g in range(len(GOAL_NAMES))])

    def reset(self, seed=None):
        with self.LOCK:
            self._env.reset()
        frame = self._render()
        for _ in range(self.frame_stack):
            self._frames.append(frame)
        self.success = np.zeros(len(GOAL_NAMES))
        return self._obs(), {"coordinates": self._xy()}

    def step(self, action):
        xy = self._xy()
        self._env.step(np.asarray(action, dtype=np.float64))
        self._frames.append(self._render())
        self._update_success()
        return self._obs(), 0.0, False, False, {"coordinates": xy, "next_coordinates": self._xy()}

    def render(self, size: int = 240) -> np.ndarray:
        return self._env.render("rgb_array", width=size, height=size)


def kitchen_metrics(success: np.ndarray) -> dict[str, float]:
    """Per-task success over all episodes, and their sum (METRA's KitchenOverall)."""
    success = np.max(success, axis=0)
    out = {f"KitchenTask{name}": float(s) for name, s in zip(GOAL_NAMES, success)}
    out["KitchenOverall"] = float(success.sum())
    return out
