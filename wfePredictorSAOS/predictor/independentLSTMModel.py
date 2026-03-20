import torch
import torch.nn as nn

class IndependentSlopeLSTM(nn.Module):
    def __init__(self, hidden_size=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (B, T, Nslopes)
        B, T, N = x.shape
        x = x.permute(0, 2, 1).reshape(B * N, T, 1)
        out, _ = self.lstm(x)
        y = self.head(out[:, -1, :])
        return y.reshape(B, N)