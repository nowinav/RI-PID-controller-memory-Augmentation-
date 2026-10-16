"""
Sweeps memory size and measures hit rate vs control error.
This is the paper figure: how does the habit layer scale with memory budget?
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
    memory_full = pickle.load(f)

all_points = memory_full.points.copy()
print(f"Full memory: {len(all_points)} points\n")

TEST_VREFS = [3.5, 4.2, 4.8, 5.3, 5.8, 6.4, 6.9]

# Memory sizes to test
MEMORY_SIZES = [100, 250, 500, 1000, 2000, 5000, 10000, len(all_points)]

# Use the medium-weight config
W1, W2, W3 = 50.0, 500.0, 1000.0

env = BuckConverterEnv()
results = []

rng = np.random.default_rng(42)

for size in MEMORY_SIZES:
    # Subsample the full memory down to 'size' points
    if size >= len(all_points):
        points = all_points
    else:
        idx = rng.choice(len(all_points), size=size, replace=False)
        points = [all_points[i] for i in idx]

    memory = HabitMemory(W1=W1, W2=W2, W3=W3,
                         min_neighbors=8, max_residual=0.05)
    memory.points = list(points)

    total_hits, total_calls = 0, 0
    mean_err, max_err = 0.0, 0.0

    for test_vref in TEST_VREFS:
        ctrl = HabitController(model, memory, use_memory=True)

        obs, _ = env.reset(seed=int(test_vref * 10))
        env.Vref = test_vref
        env.Vref_shaped = 0.0
        env.Vref_shaped_dot = 0.0
        env.prev_error = 0.0

        Vout_log = []
        for _ in range(env.n_steps):
            obs_noisy = (obs + np.random.normal(0, OBS_NOISE_STD, size=obs.shape)).astype(np.float32)
            obs_noisy = np.clip(obs_noisy, -2.0, 2.0)
            action, _ = ctrl.predict(obs_noisy, deterministic=True)
            obs, reward, term, trunc, _ = env.step(action)
            Vout_log.append(env._get("Vout"))
            if term or trunc:
                break

        s = ctrl.stats
        total_hits  += s['habit_hits']
        total_calls += s['habit_hits'] + s['rl_calls']

        ss_err = abs(np.mean(Vout_log[-100:]) - test_vref)
        mean_err += ss_err
        max_err = max(max_err, ss_err)

    mean_err /= len(TEST_VREFS)
    hit_pct = 100 * total_hits / max(total_calls, 1)

    results.append((size, hit_pct, mean_err, max_err))

    print(f"size={size:6d}  |  hit rate: {hit_pct:5.1f}%  |  "
          f"mean err: {mean_err*1000:6.2f} mV  |  max: {max_err*1000:6.2f} mV")

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sizes     = [r[0] for r in results]
hit_rates = [r[1] for r in results]
mean_errs = [r[2] * 1000 for r in results]
max_errs  = [r[3] * 1000 for r in results]

axes[0].semilogx(sizes, hit_rates, 'o-', markersize=8, lw=2)
axes[0].set_xlabel('Memory size (points)')
axes[0].set_ylabel('Hit rate (%)')
axes[0].set_title('Habit hit rate vs memory size')
axes[0].grid(True)
axes[0].set_ylim(0, 105)

axes[1].semilogx(sizes, mean_errs, 'o-', markersize=8, lw=2, label='mean')
axes[1].semilogx(sizes, max_errs,  's--', markersize=8, lw=1.5, label='max', alpha=0.6)
axes[1].set_xlabel('Memory size (points)')
axes[1].set_ylabel('Steady-state error (mV)')
axes[1].set_title('Control error vs memory size')
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()
plt.savefig('habit_memory_scaling.png', dpi=120)
plt.close()
print("\nSaved habit_memory_scaling.png")

env.close()