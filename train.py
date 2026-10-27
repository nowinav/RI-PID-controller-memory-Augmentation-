"""
Trains a TD3 agent to control the averaged buck converter with Vin variations.
"""
import os
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback
from buck_env import BuckConverterEnv

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

env = BuckConverterEnv(vin_range=(8.0, 14.0))
print(f"Env ready. Observation shape: {env.observation_space.shape}")
print(f"Training with Vin varying from {env.vin_range[0]}V to {env.vin_range[1]}V")

n_actions = env.action_space.shape[0]
action_noise = NormalActionNoise(
    mean=np.zeros(n_actions),
    sigma=0.03 * np.ones(n_actions),
)

model = TD3(
    "MlpPolicy",
    env,
    learning_rate=1e-4,
    buffer_size=200_000,
    learning_starts=5_000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    train_freq=(1, "step"),
    gradient_steps=1,
    policy_delay=2,
    target_policy_noise=0.05,
    target_noise_clip=0.1,
    action_noise=action_noise,
    policy_kwargs=dict(net_arch=[128, 128]),
    verbose=1,
    device="cpu",
    seed=42,
)

checkpoint = CheckpointCallback(
    save_freq=50_000,
    save_path=MODEL_DIR,
    name_prefix="td3_buck_robust",
)

TOTAL_STEPS = 300_000
print(f"Training for {TOTAL_STEPS} steps...\n")

try:
    model.learn(
        total_timesteps=TOTAL_STEPS,
        callback=checkpoint,
        log_interval=10,
        progress_bar=False,
    )
finally:
    env.close()

final_path = os.path.join(MODEL_DIR, "td3_buck_final")
model.save(final_path)
print(f"\nRobust model saved to {final_path}.zip")