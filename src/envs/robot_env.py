import gymnasium as gym
import numpy as np
import mujoco

class MobilePendulumEnv(gym.Env):

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path("xml/mobile_pendulum.xml")
        self.data = mujoco.MjData(self.model)

        # action = wheel torques [left, right]
        self.action_space = gym.spaces.Box(
            low=-2.0, high=2.0, shape=(2,), dtype=np.float32
        )

        # observations
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )

    def get_obs(self):
        # state vector example
        theta = self.data.qpos[2]        # pendulum joint
        theta_dot = self.data.qvel[2]
        x = self.data.qpos[0]            # chassis x
        x_dot = self.data.qvel[0]
        left_vel = self.data.qvel[0]
        right_vel = self.data.qvel[1]

        return np.array([theta, theta_dot, x, x_dot, left_vel, right_vel])

    def step(self, action):

        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)

        obs = self.get_obs()

        theta = obs[0]

        # reward: keep pendulum upright
        reward = -theta**2

        terminated = abs(theta) > np.deg2rad(45)

        return obs, reward, terminated, False, {}

    def reset(self, *, seed=None, options=None):
        mujoco.mj_resetData(self.model, self.data)
        return self.get_obs(), {}
