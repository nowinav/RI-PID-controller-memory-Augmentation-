"""
Tests the current trained policy at a fixed Vref.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import TD3
from buck_env import BuckConverterEnv

TARGET_VREF = 3.0

# Force Vref to a single value
env = BuckConverterEnv(vref_range=(TARGET_VREF, TARGET_VREF))
model = TD3.load("models/td3_buck_final")

obs, _ = env.reset(seed=42)

Vout_log, Vref_shaped_log, duty_log, t_log = [], [], [], []

for i in range(env.n_steps):
    Vref_shaped_log.append(env.Vref_shaped)
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, term, trunc, _ = env.step(action)

    Vout_log.append(env._get("Vout"))
    duty_log.append(float(action[0]))
    t_log.append(i * env.Ts * 1000)

    if term or trunc:
        break

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axes[0].plot(t_log, Vout_log, 'b-', label='Vout')
axes[0].plot(t_log, Vref_shaped_log, 'g--', label='Vref_shaped')
axes[0].axhline(TARGET_VREF, color='r', ls=':', label=f'Vref={TARGET_VREF}V')
axes[0].set_ylabel('Voltage (V)')
axes[0].legend()
axes[0].grid(True)
axes[0].set_title(f'Policy at Vref={TARGET_VREF}V')

axes[1].plot(t_log, duty_log, 'g-')
axes[1].set_ylabel('duty')
axes[1].set_xlabel('time (ms)')
axes[1].grid(True)

plt.tight_layout()
plt.savefig(f'test_vref{int(TARGET_VREF)}.png', dpi=120)
plt.close()

Vout_arr = np.array(Vout_log)
print(f"Vout final: {Vout_arr[-1]:.4f} V")
print(f"Steady-state error: {abs(Vout_arr[-1] - TARGET_VREF):.4f} V")
env.close()