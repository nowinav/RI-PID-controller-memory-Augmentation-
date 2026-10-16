"""
Gymnasium environment wrapping the averaged buck converter FMU.
Now exposes Vin for the habit memory's schema gate.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from fmpy import read_model_description, extract
from fmpy.fmi2 import FMU2Slave


class BuckConverterEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, fmu_path="BuckConverter_Averaged.fmu",
                 Tf=0.03, Ts=1e-4, Vin=12.0, n_substeps=10,
                 shaper_freq_hz=80.0,
                 vref_range=(4.8, 5.2),
                 vin_range=None):
        super().__init__()
        self.fmu_path = fmu_path
        self.Tf = Tf
        self.Ts = Ts
        self.Vin = Vin
        self.n_steps = int(round(Tf / Ts))
        self.n_substeps = n_substeps
        self.dt_sub = Ts / n_substeps
        self.vref_range = vref_range
        self.vin_range = vin_range

        self.omega_n = 2.0 * np.pi * shaper_freq_hz
        self.zeta_shaper = 1.0

        self.unzipdir = extract(fmu_path)
        self.md = read_model_description(fmu_path)
        self.vref_map = {v.name: v.valueReference for v in self.md.modelVariables}

        if self.md.coSimulation is None:
            raise RuntimeError("This FMU is Model Exchange. Re-export as Co-Simulation.")

        self.observation_space = spaces.Box(
            low=np.array([0.0, -2.0, -2.0, 0.0], dtype=np.float32),
            high=np.array([1.0,  2.0,  2.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([0.95], dtype=np.float32),
            dtype=np.float32,
        )

        self.fmu = FMU2Slave(
            guid=self.md.guid,
            unzipDirectory=self.unzipdir,
            modelIdentifier=self.md.coSimulation.modelIdentifier,
            instanceName="buck",
        )
        self.fmu.instantiate()
        self.fmu.setupExperiment(startTime=0.0)
        self.fmu.enterInitializationMode()
        self.fmu.exitInitializationMode()

        self.t = 0.0
        self.Vref = 5.0
        self.Vref_shaped = 0.0
        self.Vref_shaped_dot = 0.0
        self.step_count = 0
        self.prev_error = 0.0

    def _set(self, name, value):
        if name in self.vref_map:
            self.fmu.setReal([self.vref_map[name]], [float(value)])

    def _get(self, name):
        return float(self.fmu.getReal([self.vref_map[name]])[0])

    def _update_shaper(self):
        accel = (self.omega_n ** 2) * (self.Vref - self.Vref_shaped) \
                - 2.0 * self.zeta_shaper * self.omega_n * self.Vref_shaped_dot
        self.Vref_shaped_dot += accel * self.Ts
        self.Vref_shaped += self.Vref_shaped_dot * self.Ts
        if self.Vref_shaped > self.Vref:
            self.Vref_shaped = self.Vref
            self.Vref_shaped_dot = min(self.Vref_shaped_dot, 0.0)

    def _get_obs(self):
        Vout = self._get("Vout")
        error = self.Vref_shaped - Vout
        d_error = error - self.prev_error

        obs = np.array([
            self.Vref_shaped / self.Vin,
            error            / self.Vin,
            d_error          / self.Vin,
            Vout             / self.Vin,
        ], dtype=np.float32)
        return np.clip(obs, -2.0, 2.0).astype(np.float32)

    def _reward(self, error, Vout, d_error):
        ref_scale = max(abs(self.Vref), 1.0)
        err_rel = error / ref_scale
        tracking_cost = 3.0 * min(abs(err_rel), 1.0) ** 2
        overshoot_rel = max(0.0, Vout - self.Vref_shaped) / ref_scale
        overshoot_cost = 2.0 * min(overshoot_rel, 1.0) ** 2
        d_err_rel = d_error / ref_scale
        oscillation_cost = 5.0 * min(d_err_rel ** 2, 1.0)
        settle_bonus = 1.0 if abs(err_rel) < 0.05 else 0.0
        return -tracking_cost - overshoot_cost - oscillation_cost + settle_bonus

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        lo, hi = self.vref_range
        self.Vref = float(self.np_random.uniform(lo, hi))

        if self.vin_range is not None:
            lo_v, hi_v = self.vin_range
            self.Vin = float(self.np_random.uniform(lo_v, hi_v))
            # Requires Vin to be a runtime input port on the FMU
            self._set("Vin", self.Vin)

        self.fmu.reset()
        self.fmu.setupExperiment(startTime=0.0)
        self.fmu.enterInitializationMode()
        self.fmu.exitInitializationMode()

        self.Vref_shaped = 0.0
        self.Vref_shaped_dot = 0.0
        self.t = 0.0
        self.step_count = 0
        self._set("duty_cycle", 0.0)
        self._set("Vref", self.Vref)

        Vout0 = self._get("Vout")
        self.prev_error = self.Vref_shaped - Vout0

        return self._get_obs(), {}

    def step(self, action):
        duty = float(np.clip(action[0], 0.0, 0.95))
        self._update_shaper()
        self._set("duty_cycle", duty)
        self._set("Vref", self.Vref)
        for _ in range(self.n_substeps):
            self.fmu.doStep(currentCommunicationPoint=self.t,
                            communicationStepSize=self.dt_sub)
            self.t += self.dt_sub

        self.step_count += 1
        Vout = self._get("Vout")
        error = self.Vref_shaped - Vout
        d_error = error - self.prev_error
        self.prev_error = error

        reward = self._reward(error, Vout, d_error)
        terminated = not np.isfinite(Vout)
        truncated = self.step_count >= self.n_steps
        return self._get_obs(), reward, terminated, truncated, {}

    def close(self):
        if self.fmu is not None:
            try:
                self.fmu.terminate()
                self.fmu.freeInstance()
            except Exception:
                pass
            self.fmu = None