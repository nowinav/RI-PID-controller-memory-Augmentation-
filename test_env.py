from buck_env import BuckConverterEnv
import numpy as np

env = BuckConverterEnv()
obs, _ = env.reset()
print("Initial obs:", obs)

for i in range(200):
    # Vary duty around the nominal value
    duty = 0.35 + 0.15 * np.sin(2 * np.pi * i / 100)
    obs, reward, term, trunc, _ = env.step([duty])
    if i % 25 == 0:
        print(f"step {i:3d}  duty={duty:.3f}  Vout={obs[2]*12:.3f}V  reward={reward:+.3f}")

print("Done")
env.close()