"""Pixel-based DMC Quadruped and Humanoid with METRA's setup.

Action repeat 2, 64x64 RGB renders with position-coloured textures, 3 stacked frames as a flat uint8 vector.
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")  # must be set before mujoco is imported

from collections import deque

import numpy as np
from gymnasium import spaces

from ssd.custom_dmc_tasks import humanoid, quadruped

# name -> (task module, task name, camera id)
TASKS = {
    "dmc_quadruped": (quadruped, "run_forward_color", 2),
    "dmc_humanoid": (humanoid, "run_color", 0),
}


class DMCPixelEnv:
    size = 64
    frame_stack = 3
    action_repeat = 2
    pixel_shape = (frame_stack * 3, size, size)

    def __init__(self, name: str, seed: int):
        module, task, self._camera = TASKS[name]
        self.name = name
        self._env = module.make(task, task_kwargs={"random": seed}, environment_kwargs={"flat_observation": True})
        self.physics = self._env.physics
        self._recolor_textures()
        spec = self._env.action_spec()
        self._act_low, self._act_high = spec.minimum.astype(np.float64), spec.maximum.astype(np.float64)
        self.action_space = spaces.Box(-1.0, 1.0, shape=spec.shape, dtype=np.float32)
        self.observation_space = spaces.Box(0, 255, shape=(int(np.prod(self.pixel_shape)),), dtype=np.uint8)
        self._frames: deque = deque(maxlen=self.frame_stack)

    @property
    def unwrapped(self):
        return self

    def _recolor_textures(self):
        """Colour 2-D textures by texel position, as METRA's RenderWrapper."""
        m = self.physics.model
        for i in range(m.ntex):
            if m.tex_type[i] != 0:  # mjTEXTURE_2D
                continue
            h, w, c, adr = m.tex_height[i], m.tex_width[i], m.tex_nchannel[i], m.tex_adr[i]
            xs, ys = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
            rgb = np.stack([xs / h * 255, ys / w * 255, np.full_like(xs, 128)], axis=-1).astype(np.uint8)
            tex = m.tex_data[adr:adr + h * w * c].reshape(h, w, c)
            tex[..., :3] = rgb
        m.mat_texrepeat[:, :] = 1

    def _render(self) -> np.ndarray:
        img = self.physics.render(height=self.size, width=self.size, camera_id=self._camera)
        return img.transpose(2, 0, 1)

    def _obs(self) -> np.ndarray:
        return np.concatenate(list(self._frames), axis=0).reshape(-1)

    def _xy(self) -> np.ndarray:
        return self.physics.named.data.geom_xpos["torso", ["x", "y"]].copy()

    def reset(self, seed=None):
        self._env.reset()
        frame = self._render()
        for _ in range(self.frame_stack):
            self._frames.append(frame)
        return self._obs(), {"coordinates": self._xy()}

    def step(self, action):
        xy = self._xy()
        a = self._act_low + (np.asarray(action, dtype=np.float64) + 1.0) / 2.0 * (self._act_high - self._act_low)
        for _ in range(self.action_repeat):
            if self._env.step(a).last():
                break
        self._frames.append(self._render())
        return self._obs(), 0.0, False, False, {"coordinates": xy, "next_coordinates": self._xy()}

    def render(self, size: int = 240) -> np.ndarray:
        return self.physics.render(height=size, width=size, camera_id=self._camera)
