"""
Evaluate the trained TD3 policy. Uses the refined model if available,
otherwise falls back to the base trained model.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import TD3
from buck_env import BuckConverterEnv

# Prefer refined model, fall back to base model
CANDIDATES = [
    "models/td3_buck_refined.zip",
    "models/td3_buck_final.zip",
    "models/td3_buck_final",
]
MODEL_PATH = next((p for p in CANDIDATES if os.path.isfile(p)), None)
if MODEL_PATH is None:
    raise FileNotFoundError(
        f"No trained model found. Looked for: {CANDIDATES}"
    )
print(f"Loading model: {MODEL_PATH}")

env = BuckConverterEnv()
model = TD3.load(MODEL_PATH, env=env)

obs, _ = env.reset(seed=42)

Vout_log = []
Vref_shaped_log = []
Vref_log = []
duty_log = []
reward_log = []
t_log = []

for i in range(env.n_steps):
    Vref_log.append(env.Vref)
    Vref_shaped_log.append(env.Vref_shaped)

    action, _ = model.predict(obs, deterministic=True)
    obs, reward, term, trunc, _ = env.step(action)

    Vout_log.append(env._get("Vout"))
    duty_log.append(float(action[0]))
    reward_log.append(reward)
    t_log.append(i * env.Ts * 1000)

    if term or trunc:
        break

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axes[0].plot(t_log, Vout_log, 'b-', lw=1.8, label='Vout')
axes[0].plot(t_log, Vref_shaped_log, 'g--', lw=1.5, label='Vref_shaped')
axes[0].axhline(Vref_log[0], color='r', ls=':', lw=1.5,
                label=f'Vref={Vref_log[0]:.2f}V')
axes[0].set_ylabel('Voltage (V)')
axes[0].legend(loc='lower right')
axes[0].grid(True)
axes[0].set_title(f'Shaper + TD3 — episode reward {sum(reward_log):.1f}')

axes[1].plot(t_log, duty_log, 'g-', lw=1.5)
axes[1].set_ylabel('duty')
axes[1].grid(True)

axes[2].plot(t_log, reward_log, 'k-', lw=1.0)
axes[2].set_ylabel('reward/step')
axes[2].set_xlabel('time (ms)')
axes[2].grid(True)

plt.tight_layout()
plt.savefig('policy_response.png', dpi=120)
plt.close()

Vout_arr = np.array(Vout_log)
Vref_final = Vref_log[0]
peak = Vout_arr.max()
overshoot_pct = (peak - Vref_final) / Vref_final * 100
final_err = abs(Vout_arr[-1] - Vref_final)

# Also compute RMS ripple around Vref (excluding the initial ramp)
ramp_end = int(0.005 / env.Ts)     # skip first 5 ms
ripple = Vout_arr[ramp_end:] - Vref_final
rms_ripple = float(np.sqrt(np.mean(ripple**2)))

print(f"Episode reward:     {sum(reward_log):.2f}")
print(f"Vref (final):       {Vref_final:.4f} V")
print(f"Vout final:         {Vout_arr[-1]:.4f} V")
print(f"Vout peak:          {peak:.4f} V")
print(f"Overshoot:          {overshoot_pct:.2f} %")
print(f"Steady-state error: {final_err:.4f} V ({final_err/Vref_final*100:.2f} %)")
print(f"RMS ripple:         {rms_ripple:.4f} V")
print(f"Saved policy_response.png")

env.close()