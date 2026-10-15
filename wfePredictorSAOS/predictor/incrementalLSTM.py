"""
LSTM predictor v2: incremental prediction with per-window normalisation, shared by all the slopes (temporal only).

For the window of the last p samples of one slope, s(t-p+1..t):
    m     = s(t)                                              last sample
    sigma = max(rms(ds over the window), sigma_floor)         ds(k) = s(k) - s(k-1)
    input per step k (p-1 steps): [ds(k) / sigma, (s(k) - m) / sigma]
    s_hat(t+h) = m + sigma * f_h(input),   h in horizons      one output per horizon
By construction the prediction is equivariant in offset and amplitude (independent of r0, geometry and plate scale; only the floor
breaks it for nearly constant windows) and the gain at zero frequency is exactly 1: a constant input is predicted as itself.
"""
import torch
import torch.nn as nn
import numpy as np


class IncrementalSlopeLSTM(nn.Module):
    def __init__(self, past_horizon=24, hidden_size=32, num_layers=1, horizons=(1, 2, 3, 4), sigma_floor=1e-3):
        super().__init__()
        self.past_horizon = int(past_horizon)
        self.horizons = [int(h) for h in horizons]
        self.sigma_floor = float(sigma_floor)
        self.lstm = nn.LSTM(input_size=2, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, len(self.horizons))

    def config(self):
        return dict(past_horizon=self.past_horizon, hidden_size=self.lstm.hidden_size, num_layers=self.lstm.num_layers,
                    horizons=self.horizons, sigma_floor=self.sigma_floor)

    def forward(self, x):
        """
        x : (B, p, N) windows of raw slopes, oldest first.
        Returns the predictions (B, N, H) for every horizon and the window scale sigma (B, N).
        """
        B, p, N = x.shape
        m = x[:, -1, :]                                                       # (B, N)
        ds = x[:, 1:, :] - x[:, :-1, :]                                       # (B, p-1, N)
        sigma = torch.sqrt(torch.mean(ds ** 2, dim=1)).clamp(min=self.sigma_floor)
        feats = torch.stack([ds / sigma[:, None, :], (x[:, 1:, :] - m[:, None, :]) / sigma[:, None, :]], dim=-1)   # (B, p-1, N, 2)
        feats = feats.permute(0, 2, 1, 3).reshape(B * N, p - 1, 2)
        out, _ = self.lstm(feats)
        y = self.head(out[:, -1, :]).reshape(B, N, -1)                        # normalised increments (B, N, H)
        return m[..., None] + sigma[..., None] * y, sigma


def load_incremental_lstm(model_path, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(model_path, map_location=device)
    model = IncrementalSlopeLSTM(**ckpt['config']).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model, ckpt


class OnlineIncrementalLSTMPredictor:
    """Online use (push / ready / predict) of a trained IncrementalSlopeLSTM for one of its horizons."""

    def __init__(self, n_slopes, model_path, horizon=2, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model, _ = load_incremental_lstm(model_path, self.device)
        if horizon not in self.model.horizons:
            raise ValueError(f'OnlineIncrementalLSTMPredictor - horizon {horizon} not trained in {model_path} ({self.model.horizons})')
        self.ih = self.model.horizons.index(horizon)
        self.n_slopes = n_slopes
        self.p = self.model.past_horizon
        self.buffer = np.zeros((self.p, n_slopes), dtype=np.float32)
        self.n_pushed = 0

    def push(self, slopes):
        slopes = np.asarray(slopes, dtype=np.float32)
        if slopes.shape != (self.n_slopes,):
            raise ValueError(f'OnlineIncrementalLSTMPredictor - expected shape {(self.n_slopes,)}, received {slopes.shape}')
        self.buffer = np.roll(self.buffer, -1, axis=0)
        self.buffer[-1] = slopes
        self.n_pushed += 1

    def ready(self):
        return self.n_pushed >= self.p

    @torch.no_grad()
    def predict(self):
        x = torch.from_numpy(self.buffer).unsqueeze(0).to(self.device)        # (1, p, N)
        pred, _ = self.model(x)
        return pred[0, :, self.ih].cpu().numpy().astype(np.float64)
