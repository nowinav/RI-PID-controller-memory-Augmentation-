"""
Habit Memory: stores (Vin, Vref_shaped, error, d_error) -> duty mappings.
Query via trust ellipses + local linear regression.
Falls back to RL when the neighborhood is sparse OR regression is unreliable.
"""

import numpy as np

class HabitMemory:
    def __init__(self,
                 W1=50.0, W2=500.0, W3=1000.0, W4=100.0,
                 min_neighbors=8,
                 max_residual=0.05,
                 max_duty_std=0.05):
        # W1..W4 : trust weights for [Vin, Vref_shaped, error, d_error]
        self.W = np.array([W1, W2, W3, W4], dtype=np.float64)
        self.min_neighbors = min_neighbors
        self.max_residual  = max_residual
        self.max_duty_std  = max_duty_std
        self.points = []   # (Vin, vref_shaped, error, d_error, duty)

    def add(self, Vin, vref_shaped, error, d_error, duty):
        self.points.append((
            float(Vin),
            float(vref_shaped),
            float(error),
            float(d_error),
            float(duty),
        ))

    def __len__(self):
        return len(self.points)

    def query(self, Vin, vref_shaped, error, d_error):
        info = {'gate_passed': False, 'n_neigh': 0, 'reason': 'ok'}

        if len(self.points) < self.min_neighbors:
            info['reason'] = 'L2: empty bank'
            return None, info

        q = np.array([Vin, vref_shaped, error, d_error], dtype=np.float64)
        pts = np.asarray(self.points, dtype=np.float64)
        X = pts[:, :4]
        y = pts[:, 4]

        diff = X - q
        d2 = np.sum(self.W * diff * diff, axis=1)
        mask = d2 <= 1.0
        n_neigh = int(np.sum(mask))
        info['n_neigh'] = n_neigh

        # L2: Density
        if n_neigh < self.min_neighbors:
            info['reason'] = f'L2: only {n_neigh} neighbors'
            return None, info

        X_near = X[mask]
        y_near = y[mask]

        # L4: Novelty (bounding box)
        lo = X_near.min(axis=0)
        hi = X_near.max(axis=0)
        if np.any(q < lo) or np.any(q > hi):
            info['reason'] = 'L4: query outside bounding box'
            return None, info

        # L5: Consistency (duty variance)
        duty_std = float(np.std(y_near))
        info['duty_std'] = duty_std
        if duty_std > self.max_duty_std:
            info['reason'] = f'L5: duty std {duty_std:.4f}'
            return None, info

        # L3: Residual (regression fit)
        X_aug = np.hstack([X_near, np.ones((n_neigh, 1))])
        try:
            coef, *_ = np.linalg.lstsq(X_aug, y_near, rcond=None)
        except np.linalg.LinAlgError:
            info['reason'] = 'L3: linalg error'
            return None, info

        y_pred_train = X_aug @ coef
        residual = float(np.sqrt(np.mean((y_near - y_pred_train) ** 2)))
        info['residual'] = residual
        if residual > self.max_residual:
            info['reason'] = f'L3: residual {residual:.4f}'
            return None, info

        q_aug = np.append(q, 1.0)
        duty_pred = float(np.dot(coef, q_aug))
        duty_pred = float(np.clip(duty_pred, 0.0, 0.95))
        info['gate_passed'] = True
        return duty_pred, info


class HabitController:
    def __init__(self, policy, memory, use_memory=True):
        self.policy = policy
        self.memory = memory
        self.use_memory = use_memory
        self.reset_stats()

    def reset_stats(self):
        self.stats = {
            'habit_hits':   0,
            'habit_misses': 0,
            'rl_calls':     0,
            'gate_failures': {},
        }

    def predict(self, obs, Vin, deterministic=True):
        if self.use_memory:
            duty, info = self.memory.query(Vin, obs[0], obs[1], obs[2])
            if duty is not None:
                self.stats['habit_hits'] += 1
                return np.array([duty], dtype=np.float32), None
            else:
                self.stats['habit_misses'] += 1
                reason = info.get('reason', 'unknown')
                self.stats['gate_failures'][reason] = \
                    self.stats['gate_failures'].get(reason, 0) + 1

        action, _ = self.policy.predict(obs, deterministic=deterministic)
        self.stats['rl_calls'] += 1
        return action, None