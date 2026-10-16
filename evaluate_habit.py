"""
Evaluates the trained policy with and without the habit memory.
Uses the same observation noise as population.
"""

import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import TD3
from buck_env import BuckConverterEnv
from habit_memory import HabitMemory, HabitController

MODEL_PATH   = "models/td3_buck_final.zip"
MEMORY_PATH  = "models/habit_memory.pkl"
OBS_NOISE_STD = 0.01

model = TD3.load(MODEL_PATH)
with open(MEMORY_PATH, "rb") as f:
    memory = pickle.load(f)

print(f"Loaded memory with {len(memory)} points\n")

TEST_VREFS = [3.5, 4.2, 4.8, 5.3, 5.8, 6.4, 6.9]

totals = {'habit_hits': 0, 'rl_calls': 0, 'habit_misses': 0}
per_vref = {}

env = BuckConverterEnv()

for test_vref in TEST_VREFS:
    ctrl_mem = HabitController(model, memory, use_memory=True)

    obs, _ = env.reset(seed=int(test_vref * 10))
    env.Vref = test_vref
    env.Vref_shaped = 0.0
    env.Vref_shaped_dot = 0.0
    env.prev_error = 0.0

    Vout_mem = []
    for _ in range(env.n_steps):
        obs_noisy = (obs + np.random.normal(0, OBS_NOISE_STD, size=obs.shape)).astype(np.float32)
        obs_noisy = np.clip(obs_noisy, -2.0, 2.0)

        action, _ = ctrl_mem.predict(obs_noisy, deterministic=True)
        obs, reward, term, trunc, _ = env.step(action)
        Vout_mem.append(env._get("Vout"))
        if term or trunc:
            break

    stats = ctrl_mem.stats
    total_steps = stats['habit_hits'] + stats['rl_calls']
    hit_pct = 100 * stats['habit_hits'] / max(total_steps, 1)

    for k in totals:
        totals[k] += stats[k]

    per_vref[test_vref] = (hit_pct, Vout_mem[-1], stats)

    print(f"Vref={test_vref:4.1f}  |  "
          f"habit: {stats['habit_hits']:3d}/{total_steps:3d} ({hit_pct:5.1f}%)  |  "
          f"RL: {stats['rl_calls']:3d}  |  "
          f"Vout: {Vout_mem[-1]:.4f}")

total_steps = totals['habit_hits'] + totals['rl_calls']
overall_hit_pct = 100 * totals['habit_hits'] / max(total_steps, 1)

print(f"\n=== Overall ===")
print(f"Total steps:       {total_steps}")
print(f"Habit hits:        {totals['habit_hits']} ({overall_hit_pct:.1f}%)")
print(f"Habit misses:      {totals['habit_misses']}")
print(f"RL fallback calls: {totals['rl_calls']} ({100-overall_hit_pct:.1f}%)")
print(f"Memory size:       {len(memory)} points")

fig, axes = plt.subplots(2, 1, figsize=(10, 6))
vrefs = list(per_vref.keys())
hit_pcts = [per_vref[v][0] for v in vrefs]

axes[0].bar(vrefs, hit_pcts, width=0.4, color='steelblue')
axes[0].set_ylabel('% steps from habit memory')
axes[0].set_xlabel('Vref (V)')
axes[0].set_title(f'Habit hit rate with noise (overall: {overall_hit_pct:.1f}%)')
axes[0].grid(True, axis='y')
axes[0].set_ylim(0, 100)

axes[1].bar(vrefs, [100 - h for h in hit_pcts], width=0.4, color='coral')
axes[1].set_ylabel('% steps from RL fallback')
axes[1].set_xlabel('Vref (V)')
axes[1].grid(True, axis='y')
axes[1].set_ylim(0, 100)

plt.tight_layout()
plt.savefig('habit_hit_rate.png', dpi=120)
plt.close()
print("\nSaved habit_hit_rate.png")

env.close()