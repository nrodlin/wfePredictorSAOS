import torch
import torch.nn as nn
import numpy as np

from wfePredictorSAOS.predictor.independentLSTMModel import IndependentSlopeLSTM

# =============================================================================
# ONLINE PREDICTOR
# =============================================================================
class OnlineSlopePredictor:
    def __init__(self,
                 n_slopes,
                 model_path='best_model_IndepLSTM.pt',
                 past_horizon=8,
                 hidden_size=32,
                 device=None,
                 mean=None,
                 std=None):

        self.n_slopes = n_slopes
        self.past_horizon = past_horizon
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = IndependentSlopeLSTM(hidden_size=hidden_size, num_layers=1).to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.buffer = np.zeros((past_horizon, n_slopes), dtype=np.float32)
        self.write_idx = 0
        self.is_full = False

        self.mean = None if mean is None else np.asarray(mean, dtype=np.float32)
        self.std = None if std is None else np.asarray(std, dtype=np.float32)

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
            raise ValueError(f'Shape incorrecta: esperado {(self.n_slopes,)}, recibido {slopes.shape}')

        slopes = self._normalize(slopes)

        self.buffer[self.write_idx, :] = slopes
        self.write_idx = (self.write_idx + 1) % self.past_horizon

        if self.write_idx == 0:
            self.is_full = True

    def ready(self):
        return self.is_full

    def get_ordered_buffer(self):
        if not self.is_full:
            raise RuntimeError('El buffer todavía no está lleno')
        return np.concatenate(
            (self.buffer[self.write_idx:], self.buffer[:self.write_idx]),
            axis=0
        )

    @torch.no_grad()
    def predict(self):
        x = self.get_ordered_buffer()                           # (T, N)
        x = torch.from_numpy(x).unsqueeze(0).to(self.device)   # (1, T, N)
        y = self.model(x).squeeze(0).cpu().numpy()             # (N,)
        y = self._denormalize(y)
        return y