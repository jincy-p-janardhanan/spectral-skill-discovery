"""State-based Ant and HalfCheetah with METRA's observations, assets and normaliser.

The envs never terminate; the rollout loop enforces the horizon.
"""

from pathlib import Path

import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv

ASSETS = Path(__file__).parent / "assets"

# observation mean / std, from METRA
NORMALIZER_PRESETS = {
    "ant": (
        [0.00486117, 0.011312, 0.7022248, 0.8454677, -0.00102548, -0.00300276, 0.00311523, -0.00139029,
         0.8607109, -0.00185301, -0.8556998, 0.00343217, -0.8585605, -0.00109082, 0.8558013, 0.00278213,
         0.00618173, -0.02584622, -0.00599026, -0.00379596, 0.00526138, -0.0059213, 0.27686235, 0.00512205,
         -0.27617684, -0.0033233, -0.2766923, 0.00268359, 0.27756855],
        [0.62473416, 0.61958003, 0.1717569, 0.28629342, 0.20020866, 0.20572574, 0.34922406, 0.40098143,
         0.3114514, 0.4024826, 0.31057045, 0.40343934, 0.3110796, 0.40245822, 0.31100526, 0.81786263, 0.8166509,
         0.9870919, 1.7525449, 1.7468817, 1.8596431, 4.502961, 4.4070187, 4.522444, 4.3518476, 4.5105968,
         4.3704205, 4.5175962, 4.3704395],
    ),
    "half_cheetah": (
        [-0.07861924, -0.08627162, 0.08968642, 0.00960849, 0.02950368, -0.00948337, 0.01661406, -0.05476654,
         -0.04932635, -0.08061652, -0.05205841, 0.04500197, 0.02638421, -0.04570961, 0.03183838, 0.01736591,
         0.0091929, -0.0115027],
        [0.4039283, 0.07610687, 0.23817, 0.2515473, 0.2698137, 0.26374814, 0.32229397, 0.2896734, 0.2774097,
         0.73060024, 0.77360505, 1.5871304, 5.5405455, 6.7097645, 6.8253727, 6.3142195, 6.417641, 5.9759197],
    ),
}


class _SkillMujocoEnv(MujocoEnv):
    """Reward-free MuJoCo env with METRA's reset distribution and optional preset normalisation."""

    metadata = {"render_modes": ["human", "rgb_array", "depth_array", "rgbd_tuple"], "render_fps": 20}
    xml: str
    name: str

    def __init__(self, normalize: bool = True, reset_noise_scale: float = 0.1, **kwargs):
        self._reset_noise_scale = reset_noise_scale
        mean, std = NORMALIZER_PRESETS[self.name]
        self._obs_mean = np.asarray(mean) if normalize else 0.0
        self._obs_std = np.asarray(std) if normalize else 1.0
        obs_space = spaces.Box(-np.inf, np.inf, shape=(len(mean),), dtype=np.float64)
        super().__init__(str(ASSETS / self.xml), frame_skip=5, observation_space=obs_space, **kwargs)

    def _raw_obs(self) -> np.ndarray:
        raise NotImplementedError

    def _xy(self) -> np.ndarray:
        raise NotImplementedError

    def _get_obs(self) -> np.ndarray:
        return (self._raw_obs() - self._obs_mean) / self._obs_std

    def step(self, action):
        xy = self._xy()
        self.do_simulation(action, self.frame_skip)
        info = {"coordinates": xy, "next_coordinates": self._xy()}
        return self._get_obs(), 0.0, False, False, info

    def reset_model(self):
        noise = self._reset_noise_scale
        qpos = self.init_qpos + self.np_random.uniform(-noise, noise, size=self.model.nq)
        qvel = self.init_qvel + noise * self.np_random.standard_normal(self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_reset_info(self):
        return {"coordinates": self._xy()}


class AntEnv(_SkillMujocoEnv):
    """obs = qpos[:15] + qvel[:14], 29-D."""

    xml, name = "ant.xml", "ant"

    def _raw_obs(self):
        return np.concatenate([self.data.qpos.flat[:15], self.data.qvel.flat[:14]])

    def _xy(self):
        return self.data.qpos[:2].copy()


class HalfCheetahEnv(_SkillMujocoEnv):
    """obs = qpos + qvel, 18-D."""

    xml, name = "half_cheetah.xml", "half_cheetah"

    def _raw_obs(self):
        return np.concatenate([self.data.qpos.flat, self.data.qvel.flat])

    def _xy(self):
        return np.array([self.data.qpos[0], 0.0])


ENVS = {"ant": AntEnv, "half_cheetah": HalfCheetahEnv}


def make_env(name: str, seed: int, normalize: bool = True) -> _SkillMujocoEnv:
    env = ENVS[name](normalize=normalize)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env
