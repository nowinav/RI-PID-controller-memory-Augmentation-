# save_deployment_memory.py
import pickle
import numpy as np
from stable_baselines3 import TD3
from habit_memory import HabitMemory

with open("models/habit_memory.pkl", "rb") as f:
    full = pickle.load(f)

rng = np.random.default_rng(42)
idx = rng.choice(len(full.points), size=500, replace=False)
deploy = HabitMemory(W1=50.0, W2=500.0, W3=1000.0, min_neighbors=8, max_residual=0.05)
deploy.points = [full.points[i] for i in idx]

with open("models/habit_memory_deployed.pkl", "wb") as f:
    pickle.dump(deploy, f)

print(f"Saved deployment memory: {len(deploy)} points")
print(f"Storage estimate: {len(deploy) * 4 * 4} bytes (float32)")  # ~8 KB