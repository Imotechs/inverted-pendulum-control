from stable_baselines3 import PPO
import time
import numpy as np
import os, sys

# ---------------- PATHS ----------------
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)
from envs.env2 import MobilePendulumEnv

# Initialize environment WITHOUT render_mode for training compatibility
env = MobilePendulumEnv(render_mode=None)

# Load the model
model = PPO.load("ppo_pendulum", env=env)

FALL_LIMIT = 60 * np.pi / 180.0
timestep = 10000

# Create a single viewer instance
import mujoco.viewer
viewer = mujoco.viewer.launch_passive(env.model, env.data)

for episode in range(10):
    obs, _ = env.reset()
    episode_reward = 0
    step_count = 0

    while step_count <= timestep:
        # Get action from trained model
        action, _ = model.predict(obs, deterministic=True)
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        episode_reward += reward
        step_count += 1
        
        # Update the viewer
        viewer.sync()
        
        # Small delay to control visualization speed
        time.sleep(0.005)
        
        # Check termination conditions
        angle_too_large = abs(info["pendulum_angle"]) > FALL_LIMIT
        
        if angle_too_large or terminated:
            print(f"Episode {episode}: Steps={step_count}, Reward={episode_reward:.2f}, "
                  f"Final Angle={np.rad2deg(info['pendulum_angle']):.1f}°")
            break
    
    if step_count > timestep:
        print(f"Episode {episode}: Completed {timestep} steps, Reward={episode_reward:.2f}")

# Close the viewer properly
viewer.close()