"""
Runs the trained policy over many episodes (varying Vref) and stores
all good (well-tracked) points into the habit memory bank.
Populates with the SAME observation noise used at evaluation time,
and includes Vin in the 5-tuple schema.
"""

import os
import pickle
import numpy as np
from stable_baselines3 import TD3
from buck_env import BuckConverterEnv
from habit_memory import HabitMemory

MODEL_PATH   = "models/td3_buck_final.zip"
MEMORY_PATH  = "models/habit_memory.pkl"
N_EPISODES   = 80
REWARD_THRESHOLD = 0.5
OBS_NOISE_STD = 0.01

env = BuckConverterEnv(vref_range=(3.0, 7.0))
model = TD3.load(MODEL_PATH)

# Use the same trust weights as the memory class
memory = HabitMemory(W1=50.0, W2=500.0, W3=1000.0, W4=100.0, min_neighbors=8)

stored = 0
for ep in range(N_EPISODES):
    obs, _ = env.reset()
    for _ in range(env.n_steps):
        # Add observation noise
        obs_noisy = (obs + np.random.normal(0, OBS_NOISE_STD, size=obs.shape)).astype(np.float32)
        obs_noisy = np.clip(obs_noisy, -2.0, 2.0)

        prev_obs = obs_noisy.copy()
        action, _ = model.predict(obs_noisy, deterministic=True)
        obs, reward, term, trunc, _ = env.step(action)

        if reward >= REWARD_THRESHOLD:
            # Store the 5-tuple including Vin
            memory.add(
                Vin=env.Vin,
                vref_shaped=prev_obs[0],
                error=prev_obs[1],
                d_error=prev_obs[2],
                duty=float(action[0]),
            )
            stored += 1

        if term or trunc:
            break

    if (ep + 1) % 10 == 0:
        print(f"Episode {ep+1:3d}/{N_EPISODES}  |  memory size: {len(memory):6d}  |  stored: {stored}")

print(f"\nTotal stored: {len(memory)} points from {N_EPISODES} episodes")

with open(MEMORY_PATH, "wb") as f:
    pickle.dump(memory, f)
print(f"Saved to {MEMORY_PATH}")

env.close()