import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import TD3
from buck_env import BuckConverterEnv
from habit_memory import HabitController

# --- 1. Load the Model AND the Memory ---
model = TD3.load("models/td3_buck_final.zip")
with open("models/habit_memory.pkl", "rb") as f:
    memory = pickle.load(f)

# We wrap the model in the HabitController so it uses the memory!
ctrl_full = HabitController(model, memory, use_memory=True)

class TunedPI:
    def __init__(self, Kp=0.05, Ki=300.0, Ts=1e-4):
        self.Kp, self.Ki, self.Ts = Kp, Ki, Ts
        self.integral = 0.0
    def reset(self): self.integral = 0.0
    def predict(self, error):
        self.integral += error * self.Ts
        u = self.Kp * error + self.Ki * self.integral
        return np.array([np.clip(u, 0.0, 0.95)], dtype=np.float32)

TARGET_VREF = 5.0
DISTURBANCE_STEP = 150 # 15 ms

# --- 2. Run the FULL System (Shaper + Memory + RL) ---
env = BuckConverterEnv(vref_range=(TARGET_VREF, TARGET_VREF), vin_range=None)
obs, _ = env.reset(seed=42)
env.Vin = 12.0
env._set("Vin", env.Vin) 

Vout_rl = []
for i in range(env.n_steps):
    if i == DISTURBANCE_STEP:
        env.Vin = 8.0 # DROP VOLTAGE TO 8V!
        env._set("Vin", env.Vin)
    
    # Here we call the HabitController, which checks the memory first!
    action, _ = ctrl_full.predict(obs, env.Vin, deterministic=True)
    obs, reward, term, trunc, _ = env.step(action)
    Vout_rl.append(env._get("Vout"))
    if term or trunc: break
env.close()

# Check how many times the memory answered vs the RL
print(f"Memory Hits: {ctrl_full.stats['habit_hits']} | RL Fallbacks: {ctrl_full.stats['rl_calls']}")

# --- 3. Run the Fixed PI Agent ---
env_pi = BuckConverterEnv(vref_range=(TARGET_VREF, TARGET_VREF), vin_range=None)
obs, _ = env_pi.reset(seed=42)
env_pi._update_shaper = lambda: None
env_pi.Vref_shaped = TARGET_VREF
env_pi.Vin = 12.0
env_pi._set("Vin", env_pi.Vin)

ctrl_pi = TunedPI()
Vout_pi = []
for i in range(env_pi.n_steps):
    if i == DISTURBANCE_STEP:
        env_pi.Vin = 8.0 
        env_pi._set("Vin", env_pi.Vin)

    error = TARGET_VREF - env_pi._get("Vout")
    action = ctrl_pi.predict(error)
    env_pi._set("duty_cycle", float(action[0]))
    env_pi._set("Vref", TARGET_VREF)
    env_pi._set("Vin", env_pi.Vin)
    
    for _ in range(env_pi.n_substeps):
        env_pi.fmu.doStep(env_pi.t, env_pi.dt_sub)
        env_pi.t += env_pi.dt_sub
    Vout_pi.append(env_pi._get("Vout"))
    env_pi.step_count += 1
env_pi.close()

# --- 4. Plot the Results ---
fig, ax = plt.subplots(figsize=(10, 5))
t = np.arange(len(Vout_rl)) * 1e-4 * 1000

ax.plot(t, Vout_rl, 'b-', linewidth=2.5, label='Full Architecture (Shaper + Habit Memory + RL)')
ax.plot(t, Vout_pi, 'r--', linewidth=1.5, label='Tuned PI (Tuned for 12V)')
ax.axhline(TARGET_VREF, color='k', linestyle=':', label='Target = 5.0V')
ax.axvline(DISTURBANCE_STEP * 1e-4 * 1000, color='gray', linestyle='--', label='Line Disturbance: Vin drops 12V -> 8V')

ax.set_ylabel('Vout (V)')
ax.set_xlabel('Time (ms)')
ax.set_title('Full Architecture Deployment: Line Disturbance')
ax.legend(loc='lower right')
ax.grid(True)
plt.tight_layout()
plt.savefig('line_disturbance.png', dpi=120)
plt.close()
print("Saved line_disturbance.png")