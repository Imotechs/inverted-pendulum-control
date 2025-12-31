import gymnasium as gym
import numpy as np
import mujoco
import mujoco.viewer


class MobilePendulumEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 200}

    def __init__(self, render_mode=None):
        super().__init__()

        self.model = mujoco.MjModel.from_xml_path("xml/mobile_pendulum.xml")
        self.data = mujoco.MjData(self.model)

        # Wheel dofs
        self.left_dof = self.model.joint("left_wheel_joint").dofadr[0]
        self.right_dof = self.model.joint("right_wheel_joint").dofadr[0]

        # Pendulum joint
        self.pend_qpos = self.model.joint("pendulum_hinge").qposadr[0]
        self.pend_qvel = self.model.joint("pendulum_hinge").dofadr[0]

        # Action space: velocity commands for wheels
        self.action_space = gym.spaces.Box(
            low=-10.0, high=10.0, shape=(2,), dtype=np.float32
        )

        # Observation: pendulum angle, angular velocity, chassis velocity, wheel velocities
        # Removed unbounded chassis position
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )

        self.render_mode = render_mode
        self.viewer = None
        self.initial_chassis_x = 0.0

    def get_obs(self):
        # Pendulum angle and velocity
        theta = self.data.qpos[self.pend_qpos]
        theta = (theta + np.pi) % (2*np.pi) - np.pi
        theta_dot = self.data.qvel[self.pend_qvel]
        
        # Chassis velocity (not position)
        chassis_vel = self.data.body('chassis').cvel[0]
        
        # Wheel velocities
        left_vel = self.data.qvel[self.left_dof]
        right_vel = self.data.qvel[self.right_dof]
        
        return np.array([
            theta,           # Pendulum angle [-pi, pi]
            theta_dot,       # Pendulum angular velocity
            chassis_vel,     # Chassis x velocity (not position)
            left_vel,        # Left wheel velocity
            right_vel        # Right wheel velocity
        ], dtype=np.float32)

    def step(self, action):
        # Clip and apply actions
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:] = action
        
        # Integrate simulation (reduced to 5 for faster training)
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
        
        # Get observations
        obs = self.get_obs()
        theta = obs[0]
        theta_dot = obs[1]
        chassis_vel = obs[2]
        
        # IMPROVED REWARD FUNCTION:
        # 1. Strong penalty for angle deviation (primary goal)
        angle_reward = np.cos(theta)  # Range: [-1, 1], max at theta=0
        
        # 2. Penalty for angular velocity (encourage stability)
        velocity_penalty = 0.1 * theta_dot**2
        
        # 3. Small penalty for chassis velocity (discourage excessive movement)
        chassis_penalty = 0.01 * chassis_vel**2
        
        # 4. Action penalty (encourage smooth control)
        action_penalty = 0.001 * np.sum(np.square(action))
        
        # 5. Bonus for staying upright
        upright_bonus = 1.0 if abs(theta) < np.deg2rad(10) else 0.0
        
        # Total reward
        reward = (angle_reward 
                  - velocity_penalty 
                  - chassis_penalty 
                  - action_penalty 
                  + upright_bonus)
        
        # Termination condition (more lenient during training)
        terminated = abs(theta) > np.deg2rad(70)
        
        # Additional info for debugging
        info = {
            "pendulum_angle": theta,
            "pendulum_velocity": theta_dot,
            "chassis_velocity": chassis_vel
        }
        
        return obs, reward, terminated, False, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Random initial pendulum angle (close to upright)
        self.data.qpos[self.pend_qpos] = self.np_random.uniform(-0.15, 0.15)
        self.data.qvel[self.pend_qvel] = self.np_random.uniform(-0.05, 0.05)
        
        # Small random initial chassis position
        self.data.qpos[0] = self.np_random.uniform(-0.3, 0.3)
        
        # Store initial position
        self.initial_chassis_x = self.data.body('chassis').xpos[0]

        return self.get_obs(), {}

    def render(self):
        if self.render_mode != "human":
            return
        
        if self.viewer is None or not self.viewer.is_running():
            self.viewer = mujoco.viewer.launch_passive(
                self.model, self.data, 
                show_left_ui=False, 
                show_right_ui=False
            )
        
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None