import torch.nn as nn


class SlopesLSTM(nn.Module):
    def __init__(self, input_size=1176, hidden_size=512, num_layers=2, dropout=0.0):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        """
        x: (batch, seq_len, nSlopes)
        returns: (batch, nSlopes)
        """
        out, (h_n, c_n) = self.lstm(x)

        # Last timestep of the output
        last_out = out[:, -1, :]   # (batch, hidden_size)

        y = self.head(last_out)    # (batch, nSlopes)
        return y