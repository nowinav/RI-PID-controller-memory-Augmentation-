import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pyfmi import load_fmu
from stable_baselines3 import PPO

FMU_PATH = "/home/abhinav/Documents/RL demo files/fmu/Feedback.fmu"
STEP_SIZE = 0.1
MAX_STEPS = 100


class ModelicaEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # 1. OBSERVATION SPACE: Now 3-dimensional (error, output, derivative of output)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-10.0, high=10.0, shape=(1,), dtype=np.float32
        )
        self.fmu = None
        self.t = 0.0
        self.step_count = 0
        self.prev_y = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.fmu = load_fmu(FMU_PATH)
        self.fmu.initialize()
        self.t = 0.0
        self.step_count = 0
        self.prev_y = 0.0
        return self._get_obs(), {}

    def step(self, action):
        self.fmu.set('u', float(action[0]))
        self.fmu.do_step(current_t=self.t, step_size=STEP_SIZE)
        self.t += STEP_SIZE
        self.step_count += 1

        obs = self._get_obs()
        error = obs[0]
        
        # 2. REWARD FUNCTION: Increased action penalty from 0.01 to 0.1
        reward = -abs(error) - 0.1 * abs(float(action[0]))
        
        terminated = self.step_count >= MAX_STEPS
        return obs, reward, terminated, False, {}

    def _get_obs(self):
        try:
            error = float(self.fmu.get('feedback.y')[0])
            y = float(self.fmu.get('secondOrder.y')[0])
            
            # 3. CALCULATE DERIVATIVE: How fast is the output changing?
            der_y = (y - self.prev_y) / STEP_SIZE
            self.prev_y = y
        except Exception:
            error, y, der_y = 0.0, 0.0, 0.0
        return np.array([error, y, der_y], dtype=np.float32)


if __name__ == "__main__":
    env = ModelicaEnv()

    model = PPO("MlpPolicy", env, verbose=1, n_steps=256, batch_size=64)
    print("Training... sit back for a few minutes.")
    
    # 4. TRAIN LONGER: Increased from 20,000 to 100,000 steps
    model.learn(total_timesteps=100_000)
    
    model.save("rl_controller")
    print("Saved rl_controller.zip")
