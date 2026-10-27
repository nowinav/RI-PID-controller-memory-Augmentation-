"""
Evaluates the trained policy with and without the habit memory.
Compares against a Classical PI Controller.
Uses Conditional Integration to prevent windup.
"""

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
OBS_NOISE_STD = 0.01

model = TD3.load(MODEL_PATH)
with open(MEMORY_PATH, "rb") as f:
    memory = pickle.load(f)

print(f"Loaded memory with {len(memory)} points\n")

# --- Classical PI Controller with Anti-Windup ---
class PIController:
    def __init__(self, Kp=0.02, Ki=5.0, Ts=1e-4, u_min=0.0, u_max=0.95):
        self.Kp, self.Ki, self.Ts = Kp, Ki, Ts
        self.u_min, self.u_max = u_min, u_max
        self.integral = 0.0

    def reset(self): self.integral = 0.0

    def predict(self, error):
        p_term = self.Kp * error
        u_raw = p_term + self.integral
        
        # Anti-windup: only integrate if we aren't saturated
        if (u_raw < self.u_max and error > 0) or (u_raw > self.u_min and error < 0):
            self.integral += self.Ki * error * self.Ts
            
        u = np.clip(u_raw, self.u_min, self.u_max)
        return np.array([u], dtype=np.float32)

TEST_VREFS = [3.5, 4.2, 4.8, 5.3, 5.8, 6.4, 6.9]
PLOT_VREF = 5.3
plot_data = {}

print(f"{'Vref':>5} | {'Hit%':>6} | {'RL Calls':>8} | {'Vout (Mem)':>10} | {'Vout (RL)':>10} | {'Vout (PI)':>10}")
print("-" * 75)

for test_vref in TEST_VREFS:
    env = BuckConverterEnv(vref_range=(test_vref, test_vref))
    
    # 1. Run WITH memory (+ conditional integrator)
    ctrl_mem = HabitController(model, memory, use_memory=True)
    obs, _ = env.reset(seed=int(test_vref * 10))
    env.Vref_shaped = 0.0
    env.Vref_shaped_dot = 0.0
    
    Vout_mem = []
    residual_integral = 0.0
    for _ in range(env.n_steps):
        action, _ = ctrl_mem.predict(obs, env.Vin, deterministic=True)
        
        # CONDITIONAL INTEGRATOR: Only activate when close to target (prevents windup)
        error = env.Vref - env._get("Vout")
        if abs(error) < 0.3: 
            residual_integral += 10.0 * error * env.Ts
            action[0] = np.clip(action[0] + residual_integral, 0.0, 0.95)
        
        obs, reward, term, trunc, _ = env.step(action)
        Vout_mem.append(env._get("Vout"))
        if term or trunc: break

    # 2. Run Classical PI (Raw step)
    env_pi = BuckConverterEnv(vref_range=(test_vref, test_vref))
    obs, _ = env_pi.reset(seed=int(test_vref * 10))
    env_pi._update_shaper = lambda: None
    env_pi.Vref_shaped = test_vref
    
    ctrl_pi = PIController()
    ctrl_pi.reset()
    
    Vout_pi = []
    for _ in range(env_pi.n_steps):
        error = test_vref - env_pi._get("Vout")
        action = ctrl_pi.predict(error)
        
        env_pi._set("duty_cycle", float(action[0]))
        env_pi._set("Vref", test_vref)
        for _ in range(env_pi.n_substeps):
            env_pi.fmu.doStep(currentCommunicationPoint=env_pi.t, communicationStepSize=env_pi.dt_sub)
            env_pi.t += env_pi.dt_sub
        
        Vout_pi.append(env_pi._get("Vout"))
        env_pi.step_count += 1
        if env_pi.step_count >= env_pi.n_steps: break
            
    env.close()
    env_pi.close()

    stats = ctrl_mem.stats
    total_steps = stats['habit_hits'] + stats['rl_calls']
    hit_pct = 100 * stats['habit_hits'] / max(total_steps, 1)
    
    if test_vref == PLOT_VREF:
        plot_data = {
            'time': np.arange(len(Vout_mem)) * env.Ts * 1000,
            'Vref': test_vref,
            'mem': Vout_mem,
            'pi': Vout_pi
        }

    print(f"{test_vref:5.1f} | {hit_pct:6.1f} | {stats['rl_calls']:8d} | {Vout_mem[-1]:10.4f} | {'(off)':>10} | {Vout_pi[-1]:10.4f}")

# --- Plot 1: Vout vs Time Comparison ---
if plot_data:
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(plot_data['time'], plot_data['mem'], 'b-', linewidth=2.5, label='Our Approach (Shaper + RL + Integrator)')
    ax.plot(plot_data['time'], plot_data['pi'], 'r--', linewidth=1.5, label='Classical PI (Raw Step)')
    ax.axhline(plot_data['Vref'], color='k', linestyle=':', label=f'Target = {plot_data["Vref"]}V')
    
    ax.set_ylabel('Vout (V)')
    ax.set_xlabel('Time (ms)')
    ax.set_title(f'Controller Comparison: Vout vs Time (Vref = {plot_data["Vref"]}V)')
    ax.legend(loc='lower right')
    ax.grid(True)
    
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 6.5)
    
    plt.tight_layout()
    plt.savefig('vout_comparison.png', dpi=120)
    plt.close()
    print("\nSaved vout_comparison.png")