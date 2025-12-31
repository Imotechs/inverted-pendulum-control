from stable_baselines3 import PPO
import sys, os

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root = os.path.join(root)
sys.path.append(root)
print(root)

from envs.env2 import MobilePendulumEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

# Create vectorized environment
env = DummyVecEnv([lambda: MobilePendulumEnv(render_mode=None)])

# Create PPO model with tuned hyperparameters
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,  # Encourage exploration
    verbose=1,
    tensorboard_log="./ppo_pendulum_tensorboard/"
)

# Save checkpoints during training
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path="./models/",
    name_prefix="ppo_pendulum"
)

# Train for longer
model.learn(total_timesteps=500000, callback=checkpoint_callback)
model.save("ppo_pendulum_final")