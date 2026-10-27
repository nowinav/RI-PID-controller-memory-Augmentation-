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

class TunedPI:
    def __init__(self, Kp=0.05, Ki=300.0, Ts=1e-4):
        self.Kp, self.Ki, self.Ts = Kp, Ki, Ts
        self.integral = 0.0
    def reset(self): self.integral = 0.0
    def predict(self, error):
        self.integral += error * self.Ts
        u = self.Kp * error + self.Ki * self.integral
        return np.array([np.clip(u, 0.0, 0.95)], dtype=np.float32)

TEST_VREFS = [3.5, 5.0, 7.0]

print(f"{'Vref':>5} | {'Hit%':>6} | {'Vout (Mem)':>10} | {'Vout (PI)':>10}")
print("-" * 55)

for test_vref in TEST_VREFS:
    # FORCE Vin to 12V
    env = BuckConverterEnv(vref_range=(test_vref, test_vref), vin_range=(12.0, 12.0))
    ctrl_mem = HabitController(model, memory, use_memory=True)
    obs, _ = env.reset(seed=int(test_vref * 10))
    
    Vout_mem = []
    for _ in range(env.n_steps):
        obs_noisy = (obs + np.random.normal(0, OBS_NOISE_STD, size=obs.shape)).astype(np.float32)
        obs_noisy = np.clip(obs_noisy, -2.0, 2.0)
        action, _ = ctrl_mem.predict(obs_noisy, env.Vin, deterministic=True)
        obs, reward, term, trunc, _ = env.step(action)
        Vout_mem.append(env._get("Vout"))
        if term or trunc: break
    env.close()

    # PI Baseline
    env_pi = BuckConverterEnv(vref_range=(test_vref, test_vref), vin_range=(12.0, 12.0))
    obs, _ = env_pi.reset(seed=int(test_vref * 10))
    env_pi._update_shaper = lambda: None
    env_pi.Vref_shaped = test_vref
    
    ctrl_pi = TunedPI()
    Vout_pi = []
    for _ in range(env_pi.n_steps):
        error = test_vref - env_pi._get("Vout")
        action = ctrl_pi.predict(error)
        env_pi._set("duty_cycle", float(action[0]))
        for _ in range(env_pi.n_substeps):
            env_pi.fmu.doStep(env_pi.t, env_pi.dt_sub)
            env_pi.t += env_pi.dt_sub
        Vout_pi.append(env_pi._get("Vout"))
        env_pi.step_count += 1
    env_pi.close()

    stats = ctrl_mem.stats
    total_steps = stats['habit_hits'] + stats['rl_calls']
    hit_pct = 100 * stats['habit_hits'] / max(total_steps, 1)

    print(f"{test_vref:5.1f} | {hit_pct:6.1f} | {Vout_mem[-1]:10.4f} | {Vout_pi[-1]:10.4f}")