import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import TD3
from buck_env import BuckConverterEnv

model = TD3.load("models/td3_buck_final.zip")
TARGET_VREF = 5.3

# --- Aggressive, Fair PI Controller ---
class TunedPI:
    def __init__(self, Kp=0.05, Ki=300.0, Ts=1e-4):
        self.Kp, self.Ki, self.Ts = Kp, Ki, Ts
        self.integral = 0.0
    def reset(self): self.integral = 0.0
    def predict(self, error):
        self.integral += error * self.Ts
        u = self.Kp * error + self.Ki * self.integral
        return np.array([np.clip(u, 0.0, 0.95)], dtype=np.float32)

# 1. Run RL + Shaper + Integrator
env = BuckConverterEnv(vref_range=(TARGET_VREF, TARGET_VREF), vin_range=None)
env.Vin = 12.0
env._set("Vin", 12.0)
obs, _ = env.reset(seed=42)
Vout_rl = []
residual_integral = 0.0
for _ in range(env.n_steps):
    action, _ = model.predict(obs, deterministic=True)
    # Conditional Integrator
    error = env.Vref - env._get("Vout")
    if abs(error) < 0.5:
        residual_integral += 10.0 * error * env.Ts
        action[0] = np.clip(action[0] + residual_integral, 0.0, 0.95)
    obs, _, term, trunc, _ = env.step(action)
    Vout_rl.append(env._get("Vout"))
    if term or trunc: break
env.close()

# 2. Run Fair PI Controller
env_pi = BuckConverterEnv(vref_range=(TARGET_VREF, TARGET_VREF), vin_range=None)
env_pi.Vin = 12.0
env_pi._set("Vin", 12.0)
env_pi._update_shaper = lambda: None
env_pi.Vref_shaped = TARGET_VREF
obs, _ = env_pi.reset(seed=42)

ctrl_pi = TunedPI() # Kp=0.05, Ki=300
Vout_pi = []
for _ in range(env_pi.n_steps):
    error = TARGET_VREF - env_pi._get("Vout")
    action = ctrl_pi.predict(error)
    env_pi._set("duty_cycle", float(action[0]))
    for _ in range(env_pi.n_substeps):
        env_pi.fmu.doStep(env_pi.t, env_pi.dt_sub)
        env_pi.t += env_pi.dt_sub
    Vout_pi.append(env_pi._get("Vout"))
    env_pi.step_count += 1
env_pi.close()

# 3. Plot
fig, ax = plt.subplots(figsize=(10, 5))
t = np.arange(len(Vout_rl)) * 1e-4 * 1000
ax.plot(t, Vout_rl, 'b-', linewidth=2.5, label='Our Approach (Shaper + RL + Integrator)')
ax.plot(t, Vout_pi, 'r--', linewidth=1.5, label='Classical PI (Raw Step)')
ax.axhline(TARGET_VREF, color='k', linestyle=':', label=f'Target = {TARGET_VREF}V')

ax.set_ylabel('Vout (V)')
ax.set_xlabel('Time (ms)')
ax.set_title(f'Controller Comparison: Vout vs Time (Vref = {TARGET_VREF}V)')
ax.legend(loc='lower right')
ax.grid(True)
plt.tight_layout()
plt.savefig('vout_comparison.png', dpi=120)
plt.close()
print("Saved new, fair vout_comparison.png")