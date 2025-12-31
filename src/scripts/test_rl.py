import os
import sys
import time
import numpy as np
from stable_baselines3 import PPO

# ---------------- PATHS ----------------
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)

from envs.env2 import MobilePendulumEnv

# -------------- LOAD ENV ----------------
env = MobilePendulumEnv(render_mode="human")

# -------------- LOAD MODEL --------------
model = PPO.load("ppo_pendulum", env=env)

obs, _ = env.reset()

# upright fail limit (radians)
FALL_LIMIT = 25 * np.pi / 180.0

while True:
    action, _ = model.predict(obs, deterministic=True)

    obs, reward, terminated, truncated, info = env.step(action)

    env.render()

    # optional: slow down to real time
    time.sleep(0.01)

    # stop if pendulum falls
    pend_angle = info.get("pendulum_angle", 0.0)
    if abs(pend_angle) > FALL_LIMIT:
        print("Pendulum fell. Resetting...")
        obs, _ = env.reset()

    if terminated or truncated:
        obs, _ = env.reset()
