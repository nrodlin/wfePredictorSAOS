import torch
import numpy as np

class OnlineLinearSlopePredictor:
    def __init__(self,
                 n_slopes,
                 past_horizon=8,
                 steps_ahead=2,
                 device=None,
                 mean=None,
                 std=None):

        self.n_slopes = n_slopes
        self.past_horizon = past_horizon
        self.steps_ahead = steps_ahead
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.buffer = np.zeros((past_horizon, n_slopes), dtype=np.float32)
        self.write_idx = 0
        self.is_full = False

        self.mean = None if mean is None else np.asarray(mean, dtype=np.float32)
        self.std = None if std is None else np.asarray(std, dtype=np.float32)
        
        # Precompute the linear fitting weights safely
        x = np.arange(past_horizon)
        target_x = past_horizon - 1 + steps_ahead
        X = np.vstack([x, np.ones(len(x))]).T
        X_pred = np.array([target_x, 1])
        # weights = X_pred @ (X^T X)^-1 X^T
        self.weights = X_pred @ np.linalg.pinv(X) # Shape: (past_horizon,)

    def _normalize(self, x):
        if self.mean is None or self.std is None:
            return x
        return (x - self.mean) / np.clip(self.std, 1e-6, None)

    def _denormalize(self, x):
        if self.mean is None or self.std is None:
            return x
        return x * self.std + self.mean

    def push(self, slopes):
        slopes = np.asarray(slopes, dtype=np.float32)

        if slopes.shape != (self.n_slopes,):
            raise ValueError(f'Incorrect shape: expected {(self.n_slopes,)}, received {slopes.shape}')

        slopes = self._normalize(slopes)

        self.buffer[self.write_idx, :] = slopes
        self.write_idx = (self.write_idx + 1) % self.past_horizon

        if self.write_idx == 0:
            self.is_full = True

    def ready(self):
        return self.is_full

    def get_ordered_buffer(self):
        if not self.is_full:
            raise RuntimeError('Buffer is not completely full yet')
        return np.concatenate(
            (self.buffer[self.write_idx:], self.buffer[:self.write_idx]),
            axis=0
        )

    def predict(self):
        x = self.get_ordered_buffer() # Shape: (T, N)
        # Apply the precomputed linear weights across the time axis (axis 0)
        y = self.weights @ x          # Shape: (N,)
        y = self._denormalize(y)
        return y
