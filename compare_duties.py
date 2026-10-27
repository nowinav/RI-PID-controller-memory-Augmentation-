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

# FORCE Vin to 12V
env = BuckConverterEnv(vref_range=(TARGET_VREF, TARGET_VREF), vin_range=(12.0, 12.0))
obs, _ = env.reset(seed=42)

rl_actions = []
mem_actions = []

for step in range(env.n_steps):
    obs_noisy = (obs + np.random.normal(0, OBS_NOISE_STD, size=obs.shape)).astype(np.float32)
    obs_noisy = np.clip(obs_noisy, -2.0, 2.0)

    rl_action_arr, _ = model.predict(obs_noisy, deterministic=True)
    rl_action = float(rl_action_arr[0])

    mem_action, info = memory.query(env.Vin, obs_noisy[0], obs_noisy[1], obs_noisy[2])
    
    if mem_action is not None:
        mem_actions.append(mem_action)
    else:
        mem_actions.append(np.nan)

    rl_actions.append(rl_action)
    obs, reward, term, trunc, _ = env.step([rl_action])
    if term or trunc: break

env.close()

rl_actions = np.array(rl_actions)
mem_actions = np.array(mem_actions)
valid_mask = ~np.isnan(mem_actions)

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
t = np.arange(len(rl_actions))

axes[0].plot(t, rl_actions, 'k-', label='RL Policy Output', linewidth=2)
axes[0].plot(t, mem_actions, 'go', label='Habit Memory Output', markersize=4, alpha=0.6)
axes[0].set_ylabel('Duty Cycle (0 to 1)')
axes[0].legend()
axes[0].grid(True)

# PROTECTED ERROR CALCULATION
if np.sum(valid_mask) > 0:
    errors = np.abs(rl_actions[valid_mask] - mem_actions[valid_mask])
    axes[1].plot(t[valid_mask], errors, 'r.', alpha=0.6)
    print(f"Memory Hits: {np.sum(valid_mask)} ({100*np.mean(valid_mask):.1f}%)")
    print(f"Mean Duty Error: {np.mean(errors):.5f}")
else:
    print("No memory hits to compare.")
    
axes[1].axhline(0, color='k', linestyle='--', alpha=0.5)
axes[1].set_ylabel('Absolute Error (|Mem - RL|)')
axes[1].set_xlabel('Control Step (100 µs)')
axes[1].grid(True)

plt.tight_layout()
plt.savefig('duty_comparison.png', dpi=120)
print("Saved duty_comparison.png")