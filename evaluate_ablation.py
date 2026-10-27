import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import TD3
from buck_env import BuckConverterEnv

model = TD3.load("models/td3_buck_final.zip")
TARGET_VREF = 5.0

# 1. With Shaper
env1 = BuckConverterEnv(vref_range=(TARGET_VREF, TARGET_VREF))
obs, _ = env1.reset(seed=42)
Vout_shaped = []
for _ in range(env1.n_steps):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, term, trunc, _ = env1.step(action)
    Vout_shaped.append(env1._get("Vout"))
    if term or trunc: break
env1.close()

# 2. Without Shaper (Raw step)
env2 = BuckConverterEnv(vref_range=(TARGET_VREF, TARGET_VREF))
obs, _ = env2.reset(seed=42)
env2._update_shaper = lambda: None
env2.Vref_shaped = TARGET_VREF
Vout_raw = []
for _ in range(env2.n_steps):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, term, trunc, _ = env2.step(action)
    Vout_raw.append(env2._get("Vout"))
    if term or trunc: break
env2.close()

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
t = np.arange(len(Vout_shaped)) * 1e-4 * 1000
ax.plot(t, Vout_shaped, 'b-', linewidth=2, label='RL + Shaper')
ax.plot(t, Vout_raw, 'r--', linewidth=1.5, label='RL + Raw Step (No Shaper)')
ax.axhline(TARGET_VREF, color='k', linestyle=':')
ax.set_ylabel('Vout (V)')
ax.set_xlabel('Time (ms)')
ax.set_title('Ablation: Necessity of Reference Shaper')
ax.legend()
ax.grid(True)
plt.savefig('ablation_shaper.png', dpi=120)
print("Saved ablation_shaper.png. Peak without shaper:", max(Vout_raw))