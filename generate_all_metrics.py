import json
import pickle
import numpy as np
from stable_baselines3 import TD3
from buck_env import BuckConverterEnv

# Load the FINAL retrained model
model = TD3.load("models/td3_buck_final.zip")

class TunedPI:
    def __init__(self, Kp=0.05, Ki=300.0, Ts=1e-4):
        self.Kp, self.Ki, self.Ts = Kp, Ki, Ts
        self.integral = 0.0
    def reset(self): self.integral = 0.0
    def predict(self, error):
        self.integral += error * self.Ts
        u = self.Kp * error + self.Ki * self.integral
        return np.array([np.clip(u, 0.0, 0.95)], dtype=np.float32)

def compute_metrics(Vout, Vref, Ts, dist_step=None):
    Vout = np.array(Vout)
    time = np.arange(len(Vout)) * Ts * 1000 
    
    peak = Vout.max()
    overshoot = (peak - Vref) / Vref * 100 if peak > Vref else 0.0
    ss_err = abs(np.mean(Vout[-20:]) - Vref) / Vref * 100
    
    max_drop = 0.0
    if dist_step is not None and dist_step < len(Vout):
        max_drop = Vref - np.min(Vout[dist_step:])
        
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
        "final_vout": round(float(Vout[-1]), 3),
        "max_drop_V": round(float(max_drop), 3)
    }

results = {}
TARGET_VREF = 5.0
DIST_STEP = 150

print("Running Nominal Test (12V)...")
# 1. RL Nominal
env = BuckConverterEnv(vref_range=(TARGET_VREF, TARGET_VREF), vin_range=(12.0, 12.0))
obs, _ = env.reset(seed=42)
Vout_rl = []
for _ in range(env.n_steps):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, term, trunc, _ = env.step(action)
    Vout_rl.append(env._get("Vout"))
    if term or trunc: break
env.close()

# 2. PI Nominal
env_pi = BuckConverterEnv(vref_range=(TARGET_VREF, TARGET_VREF), vin_range=(12.0, 12.0))
obs, _ = env_pi.reset(seed=42)
env_pi._update_shaper = lambda: None
env_pi.Vref_shaped = TARGET_VREF
ctrl_pi = TunedPI()
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

results["Nominal_12V"] = {
    "RL_Shaper_Memory": compute_metrics(Vout_rl, TARGET_VREF, 1e-4),
    "Tuned_PI_Raw_Step": compute_metrics(Vout_pi, TARGET_VREF, 1e-4)
}

print("Running Line Disturbance Test (12V -> 8V)...")
# 3. RL Disturbance
env = BuckConverterEnv(vref_range=(TARGET_VREF, TARGET_VREF), vin_range=None)
obs, _ = env.reset(seed=42)
env.Vin = 12.0
env._set("Vin", 12.0)
Vout_rl_dist = []
for i in range(env.n_steps):
    if i == DIST_STEP:
        env.Vin = 8.0
        env._set("Vin", 8.0)
    action, _ = model.predict(obs, deterministic=True)
    obs, _, term, trunc, _ = env.step(action)
    Vout_rl_dist.append(env._get("Vout"))
    if term or trunc: break
env.close()

# 4. PI Disturbance
env_pi = BuckConverterEnv(vref_range=(TARGET_VREF, TARGET_VREF), vin_range=None)
obs, _ = env_pi.reset(seed=42)
env_pi._update_shaper = lambda: None
env_pi.Vref_shaped = TARGET_VREF
env_pi.Vin = 12.0
env_pi._set("Vin", 12.0)
ctrl_pi.reset()
Vout_pi_dist = []
for i in range(env_pi.n_steps):
    if i == DIST_STEP:
        env_pi.Vin = 8.0
        env_pi._set("Vin", 8.0)
    error = TARGET_VREF - env_pi._get("Vout")
    action = ctrl_pi.predict(error)
    env_pi._set("duty_cycle", float(action[0]))
    for _ in range(env_pi.n_substeps):
        env_pi.fmu.doStep(env_pi.t, env_pi.dt_sub)
        env_pi.t += env_pi.dt_sub
    Vout_pi_dist.append(env_pi._get("Vout"))
    env_pi.step_count += 1
env_pi.close()

results["Disturbance_12V_to_8V"] = {
    "RL_Shaper_Memory": compute_metrics(Vout_rl_dist, TARGET_VREF, 1e-4, dist_step=DIST_STEP),
    "Tuned_PI_Raw_Step": compute_metrics(Vout_pi_dist, TARGET_VREF, 1e-4, dist_step=DIST_STEP)
}

print("Running Shaper Ablation...")
# 5. RL No Shaper
env = BuckConverterEnv(vref_range=(TARGET_VREF, TARGET_VREF), vin_range=(12.0, 12.0))
obs, _ = env.reset(seed=42)
env._update_shaper = lambda: None
env.Vref_shaped = TARGET_VREF
Vout_rl_noshaper = []
for _ in range(env.n_steps):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, term, trunc, _ = env.step(action)
    Vout_rl_noshaper.append(env._get("Vout"))
    if term or trunc: break
env.close()

results["Ablation_No_Shaper"] = compute_metrics(Vout_rl_noshaper, TARGET_VREF, 1e-4)

with open("final_metrics.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nUpdated final_metrics.json with the absolute latest data!")