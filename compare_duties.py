"""
Compares the duty cycle predicted by the Habit Memory vs. the RL Policy
at every step of an episode. Generates duty_comparison.png.
"""

import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import TD3
from buck_env import BuckConverterEnv

MODEL_PATH   = "models/td3_buck_final.zip"
MEMORY_PATH  = "models/habit_memory.pkl"
OBS_NOISE_STD = 0.01
TARGET_VREF  = 5.0

model = TD3.load(MODEL_PATH)
with open(MEMORY_PATH, "rb") as f:
    memory = pickle.load(f)

env = BuckConverterEnv(vref_range=(TARGET_VREF, TARGET_VREF))
obs, _ = env.reset(seed=42)

rl_actions = []
mem_actions = []
hits = []

for step in range(env.n_steps):
    # 1. Add noise to observation (same as evaluation)
    obs_noisy = (obs + np.random.normal(0, OBS_NOISE_STD, size=obs.shape)).astype(np.float32)
    obs_noisy = np.clip(obs_noisy, -2.0, 2.0)

    # 2. Get RL prediction
    rl_action_arr, _ = model.predict(obs_noisy, deterministic=True)
    rl_action = float(rl_action_arr[0])

    # 3. Get Memory prediction
    # memory.query returns (duty, info_dict). We pass env.Vin for the schema gate.
    mem_action, info = memory.query(env.Vin, obs_noisy[0], obs_noisy[1], obs_noisy[2])
    
    if mem_action is not None:
        hits.append(True)
        mem_actions.append(mem_action)
    else:
        hits.append(False)
        mem_actions.append(np.nan) # Mark as NaN if memory refused to answer

    rl_actions.append(rl_action)

    # 4. Step the environment using the RL action (so the trajectory is clean)
    obs, reward, term, trunc, _ = env.step([rl_action])
    if term or trunc:
        break

env.close()

# --- Calculate Errors (only where Memory actually answered) ---
rl_actions = np.array(rl_actions)
mem_actions = np.array(mem_actions)
valid_mask = ~np.isnan(mem_actions)

if np.sum(valid_mask) > 0:
    errors = np.abs(rl_actions[valid_mask] - mem_actions[valid_mask])
    print(f"Total Steps:      {len(rl_actions)}")
    print(f"Memory Hits:      {np.sum(valid_mask)} ({100*np.mean(valid_mask):.1f}%)")
    print(f"Mean Duty Error:  {np.mean(errors):.5f}")
    print(f"Max Duty Error:   {np.max(errors):.5f}")
else:
    print("No memory hits to compare.")

# --- Plotting ---
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
t = np.arange(len(rl_actions))

# Top plot: Time series comparison
axes[0].plot(t, rl_actions, 'k-', label='RL Policy Output', linewidth=2)
axes[0].plot(t, mem_actions, 'go', label='Habit Memory Output', markersize=4, alpha=0.6)
axes[0].set_ylabel('Duty Cycle (0 to 1)')
axes[0].set_title(f'Habit Memory vs RL Policy Decisions (Vref={TARGET_VREF}V)')
axes[0].legend()
axes[0].grid(True)

# Bottom plot: Error over time
axes[1].plot(t[valid_mask], errors, 'r.', alpha=0.6)
axes[1].axhline(0, color='k', linestyle='--', alpha=0.5)
axes[1].set_ylabel('Absolute Error (|Mem - RL|)')
axes[1].set_xlabel('Control Step (100 µs)')
axes[1].set_title('Prediction Error (Memory vs RL)')
axes[1].grid(True)

plt.tight_layout()
plt.savefig('duty_comparison.png', dpi=120)
plt.close()
print("\nSaved duty_comparison.png")