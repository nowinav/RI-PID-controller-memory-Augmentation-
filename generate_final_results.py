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
OBS_NOISE_STD = 0.0 
DISTURBANCE_STEP = 150  # Inject disturbance at t = 15 ms (100 us per step)
DISTURBANCE_MAG  = 0.5  # Drop voltage by 0.5V

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
        return np.array([np.clip(u, 0.0, 0.95)], dtype=np.float32)

def inject_disturbance(env):
    """Manually drops the capacitor voltage to simulate a load step."""
    current_vout = env._get("Vout")
    env.fmu.setReal([env.vref_map['capacitor.v']], [current_vout - DISTURBANCE_MAG])

def compute_metrics(Vout, Vref, Ts):
    Vout = np.array(Vout)
    time = np.arange(len(Vout)) * Ts * 1000 
    
    # Calculate steady-state error AFTER the disturbance
    ss_err = abs(np.mean(Vout[-20:]) - Vref) / Vref * 100
    
    # Calculate max voltage drop after disturbance
    min_vout_post_dist = np.min(Vout[DISTURBANCE_STEP:])
    max_drop = Vref - min_vout_post_dist

    return {
        "final_vout": round(float(Vout[-1]), 3),
        "ss_error_pct": round(float(ss_err), 2),
        "max_drop_after_disturbance_V": round(float(max_drop), 3)
    }

# ==========================================
# 1. Run RL + Shaper (WITH Disturbance)
# ==========================================
env = BuckConverterEnv(vref_range=(TARGET_VREF, TARGET_VREF))
ctrl_mem = HabitController(model, memory, use_memory=True)
obs, _ = env.reset(seed=42)
Vout_rl = []

for i in range(env.n_steps):
    if i == DISTURBANCE_STEP:
        inject_disturbance(env)
        
    action, _ = ctrl_mem.predict(obs, env.Vin, deterministic=True)
    obs, reward, term, trunc, _ = env.step(action)
    Vout_rl.append(env._get("Vout"))
    if term or trunc: break
env.close()

# ==========================================
# 2. Run PI (WITH Disturbance)
# ==========================================
env_pi = BuckConverterEnv(vref_range=(TARGET_VREF, TARGET_VREF))
obs, _ = env_pi.reset(seed=42)
env_pi._update_shaper = lambda: None
env_pi.Vref_shaped = TARGET_VREF
ctrl_pi = TunedPI()
Vout_pi = []

for i in range(env_pi.n_steps):
    if i == DISTURBANCE_STEP:
        inject_disturbance(env_pi)

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

# ==========================================
# 3. Plotting & Saving
# ==========================================
fig, ax = plt.subplots(figsize=(10, 5))
t = np.arange(len(Vout_rl)) * 1e-4 * 1000

ax.plot(t, Vout_rl, 'b-', linewidth=2.5, label='Our Approach (Shaper + RL)')
ax.plot(t, Vout_pi, 'r--', linewidth=1.5, label='Tuned PI (Raw Step)')
ax.axhline(TARGET_VREF, color='k', linestyle=':', label=f'Target = {TARGET_VREF}V')
ax.axvline(DISTURBANCE_STEP * 1e-4 * 1000, color='gray', linestyle='--', label='Load Disturbance (-0.5V)')

ax.set_ylabel('Vout (V)')
ax.set_xlabel('Time (ms)')
ax.set_title('Fair Controller Comparison: Recovery from Load Disturbance')
ax.legend(loc='lower right')
ax.grid(True)
plt.tight_layout()
plt.savefig('disturbance_comparison.png', dpi=120)
plt.close()

results = {
    "Vref_5.0V_With_Disturbance": {
        "RL_Shaper_Memory": compute_metrics(Vout_rl, TARGET_VREF, 1e-4),
        "Tuned_PI_Raw_Step": compute_metrics(Vout_pi, TARGET_VREF, 1e-4)
    }
}
with open("final_metrics.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nSaved fair disturbance_comparison.png and final_metrics.json")
print(json.dumps(results, indent=4))