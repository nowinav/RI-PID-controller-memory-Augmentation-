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
TARGET_VREF  = 5.0
OBS_NOISE_STD = 0.01

# 1. Load Models
model = TD3.load(MODEL_PATH)
with open(MEMORY_PATH, "rb") as f:
    memory = pickle.load(f)

# 2. Define Claude's Tuned PI Controller
class TunedPI:
    def __init__(self, Kp=0.05, Ki=300.0, Ts=1e-4):
        self.Kp, self.Ki, self.Ts = Kp, Ki, Ts
        self.integral = 0.0

    def reset(self): self.integral = 0.0

    def predict(self, error):
        self.integral += error * self.Ts
        u = self.Kp * error + self.Ki * self.integral
        return np.array([np.clip(u, 0.0, 0.95)], dtype=np.float32)

def compute_metrics(Vout, Vref, Ts):
    Vout = np.array(Vout)
    time = np.arange(len(Vout)) * Ts * 1000 # ms
    
    peak = Vout.max()
    overshoot = (peak - Vref) / Vref * 100 if peak > Vref else 0.0
    
    ss_err = abs(Vout[-1] - Vref) / Vref * 100
    
    # Settling time (2% band)
    band = 0.02 * Vref
    inside = np.abs(Vout - Vref) < band
    settle_time = -1
    for i in range(len(inside)):
        if np.all(inside[i:]):
            settle_time = time[i]
            break
            
    return {
        "overshoot_pct": round(float(overshoot), 2),
        "ss_error_pct": round(float(ss_err), 2),
        "settle_time_ms": round(float(settle_time), 2),
        "final_vout": round(float(Vout[-1]), 3)
    }

# 3. Run RL + Shaper
env = BuckConverterEnv(vref_range=(TARGET_VREF, TARGET_VREF))
ctrl_mem = HabitController(model, memory, use_memory=True)
obs, _ = env.reset(seed=42)

Vout_rl = []
for _ in range(env.n_steps):
    obs_noisy = (obs + np.random.normal(0, OBS_NOISE_STD, size=obs.shape)).astype(np.float32)
    action, _ = ctrl_mem.predict(obs_noisy, env.Vin, deterministic=True)
    obs, reward, term, trunc, _ = env.step(action)
    Vout_rl.append(env._get("Vout"))
    if term or trunc: break
env.close()

# 4. Run PI (Raw Step)
env_pi = BuckConverterEnv(vref_range=(TARGET_VREF, TARGET_VREF))
obs, _ = env_pi.reset(seed=42)
env_pi._update_shaper = lambda: None  # Bypass shaper
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

# 5. Save Metrics
results = {
    "RL_Shaper_Memory": compute_metrics(Vout_rl, TARGET_VREF, 1e-4),
    "Tuned_PI_Raw_Step": compute_metrics(Vout_pi, TARGET_VREF, 1e-4)
}
with open("final_metrics.json", "w") as f:
    json.dump(results, f, indent=4)

print(json.dumps(results, indent=4))

# 6. Plot
fig, ax = plt.subplots(figsize=(10, 5))
t_rl = np.arange(len(Vout_rl)) * 1e-4 * 1000
t_pi = np.arange(len(Vout_pi)) * 1e-4 * 1000

ax.plot(t_rl, Vout_rl, 'b-', linewidth=2.5, label='Our Approach (Shaper + RL + Memory)')
ax.plot(t_pi, Vout_pi, 'r--', linewidth=1.5, label='Tuned PI (Raw Step)')
ax.axhline(TARGET_VREF, color='k', linestyle=':', label=f'Target = {TARGET_VREF}V')

ax.set_ylabel('Vout (V)')
ax.set_xlabel('Time (ms)')
ax.set_title(f'Controller Comparison (Vin = {env.Vin}V)')
ax.legend(loc='lower right')
ax.grid(True)
plt.tight_layout()
plt.savefig('vout_comparison.png', dpi=120)
plt.close()
print("Saved vout_comparison.png and final_metrics.json")