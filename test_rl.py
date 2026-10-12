import numpy as np
import matplotlib.pyplot as plt
from train_rl import ModelicaEnv
from stable_baselines3 import PPO

# Load the environment and the trained brain
env = ModelicaEnv()
model = PPO.load("rl_controller")

# Run one episode
obs, _ = env.reset()
errors = []
outputs = []
times = []

print("Testing the RL controller...")
for i in range(100):
    # The agent looks at the state and decides on an action
    action, _ = model.predict(obs, deterministic=True)
    
    # Step the simulation
    obs, reward, done, trunc, _ = env.step(action)
    
    # Save data for plotting
    errors.append(obs[0])   # The error
    outputs.append(obs[1])  # The plant output
    times.append(i * 0.1)   # Time in seconds
    
    if done:
        break

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(times, outputs, label='Plant Output (RL)', linewidth=2)
plt.axhline(y=1.0, color='r', linestyle='--', label='Setpoint (Target = 1.0)')
plt.plot(times, errors, label='Error', alpha=0.5)
plt.xlabel('Time (s)')
plt.ylabel('Value')
plt.title('RL Controller Tracking Performance')
plt.legend()
plt.grid(True)
plt.savefig('rl_performance.png')
plt.show()

print(f"Final Error: {errors[-1]:.4f}")
print("Plot saved as rl_performance.png")
