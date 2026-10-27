import json
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import TD3
from buck_env import BuckConverterEnv
from habit_memory import HabitController

MODEL_PATH   = "models/td3_buck_final.zip"
MEMORY_PATH  = "models/habit_memory.pkl"
TARGET_VREF  = 8.0  # Aggressive step to expose PI windup
OBS_NOISE_STD = 0.0 

model = TD3.load(MODEL_PATH)
with open(MEMORY_PATH, "rb") as f:
    memory = pickle.load(f)

class TunedPI:
    def __init__(self, Kp=0.05, Ki=300.0, Ts=1e-4):
        self.Kp, self.Ki, self.Ts = Kp, Ki, Ts
        self.integral = 0.0

    def reset(self): self.integral = 0.0

    def predict(self, error):
        self.integral += error * self.Ts
        u = self.Kp * error + self.Ki * self.integral
        # Notice: No anti-windup here. The integrator runs wild.
        return np.array([np.clip(u, 0.0, 0.95)], dtype=np.float32)

# --- 1. Run RL + Shaper ---
env = BuckConverterEnv(vref_range=(TARGET_VREF, TARGET_VREF))
ctrl_mem = HabitController(model, memory, use_memory=True)
obs, _ = env.reset(seed=42)
Vout_rl = []
for _ in range(env.n_steps):
    action, _ = ctrl_mem.predict(obs, env.Vin, deterministic=True)
    obs, reward, term, trunc, _ = env.step(action)
    Vout_rl.append(env._get("Vout"))
    if term or trunc: break
env.close()

# --- 2. Run PI (Raw Step) ---
env_pi = BuckConverterEnv(vref_range=(TARGET_VREF, TARGET_VREF))
obs, _ = env_pi.reset(seed=42)
env_pi._update_shaper = lambda: None # Bypass shaper, PI gets the raw 8V step
env_pi.Vref_shaped = TARGET_VREF

ctrl_pi = TunedPI()
Vout_pi = []
for _ in range(env_pi.n_steps):
    error = TARGET_VREF - env_pi._get("Vout")
    action = ctrl_pi.predict(error)
    
    env_pi._set("duty_cycle", float(action[0]))
    env_pi._set("Vref", TARGET_VREF)
    for _ in range(env_pi.n_substeps):
        env_pi.fmu.doStep(env_pi.t, env_pi.dt_sub)
        env_pi.t += env_pi.dt_sub
    
    Vout_pi.append(env_pi._get("Vout"))
    env_pi.step_count += 1
env_pi.close()

# --- 3. Plotting ---
fig, ax = plt.subplots(figsize=(10, 5))
t = np.arange(len(Vout_rl)) * 1e-4 * 1000
ax.plot(t, Vout_rl, 'b-', linewidth=2.5, label='Our Approach (Shaper + RL)')
ax.plot(t, Vout_pi, 'r--', linewidth=1.5, label='Tuned PI (Raw Step)')
ax.axhline(TARGET_VREF, color='k', linestyle=':', label=f'Target = {TARGET_VREF}V')

ax.set_ylabel('Vout (V)')
ax.set_xlabel('Time (ms)')
ax.set_title(f'Large Reference Step to {TARGET_VREF}V (Exposing PI Windup)')
ax.legend(loc='lower right')
ax.grid(True)
plt.tight_layout()
plt.savefig('large_step_comparison.png', dpi=120)
plt.close()

# --- 4. Calculate and Print Metrics ---
peak_pi = np.max(Vout_pi)
peak_rl = np.max(Vout_rl)
overshoot_pi = (peak_pi - TARGET_VREF) / TARGET_VREF * 100
overshoot_rl = (peak_rl - TARGET_VREF) / TARGET_VREF * 100
ss_err_pi = abs(Vout_pi[-1] - TARGET_VREF) / TARGET_VREF * 100
ss_err_rl = abs(Vout_rl[-1] - TARGET_VREF) / TARGET_VREF * 100

metrics = {
    "Target_Vref": TARGET_VREF,
    "RL_Shaper": {"overshoot_pct": round(overshoot_rl, 2), "final_vout": round(Vout_rl[-1], 3), "ss_error_pct": round(ss_err_rl, 2)},
    "Tuned_PI": {"overshoot_pct": round(overshoot_pi, 2), "final_vout": round(Vout_pi[-1], 3), "ss_error_pct": round(ss_err_pi, 2)}
}

with open("large_step_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\nSaved large_step_comparison.png and large_step_metrics.json")
print(json.dumps(metrics, indent=4))