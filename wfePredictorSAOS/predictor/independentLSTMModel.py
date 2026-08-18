import torch
import torch.nn as nn

class IndependentSlopeLSTM(nn.Module):
    def __init__(self, n_axis=1, hidden_size=32, num_layers=1):
        """
        n_axis: int
            Number of axis of the slopes to be predicted at the same time. Expected 1 or 2 (X|Y or X-Y)
        hidden_size: int
            Number of internal states - memory length
        num_layers: int
            Number of LSTM sequentially
        """
        super().__init__()

        if n_axis not in {1, 2}:
            raise ValueError('IndependentSlopeLSTM::__init__ - Slopes in WFS only have one or two axis.')
        else:
            # We need to store the number of axis to be predicted simultaneously to inform the forward method
            self.n_axis = n_axis
        
        self.lstm = nn.LSTM(
            input_size=n_axis,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, self.n_axis)

    def forward(self, x):
        # x is received as (time, nSlopes) or (batch_size, time, nSlopes) being nSlopes sorted as [x, y]
        is_batched = (x.dim() == 3)
        if not is_batched:
            # (time, nSlopes) -> (1, time, nSlopes)
            x = x.unsqueeze(0)

        batch_size, tSamples, nSlopes = x.shape
        n_subaps = nSlopes // self.n_axis
        
        if self.n_axis == 2:
            # (batch_size, tSamples, nSlopes) -> (batch_size, tSamples, 2, n_subaps)
            # -> permute to (batch_size, n_subaps, tSamples, 2)
            # -> reshape to (batch_size * n_subaps, tSamples, 2)
            x_sorted = (
                x.view(batch_size, tSamples, self.n_axis, n_subaps)
                .permute(0, 3, 1, 2)
                .contiguous()
                .view(batch_size * n_subaps, tSamples, self.n_axis)
            )
        else:
            # (batch_size, tSamples, nSlopes) -> permute to (batch_size, nSlopes, tSamples)
            # -> reshape to (batch_size * nSlopes, tSamples, 1)
            x_sorted = (
                x.permute(0, 2, 1)
                .contiguous()
                .view(batch_size * nSlopes, tSamples, 1)
            )

        # LSTM expects input data in the shape (batch, time, feature) = (batch_size * n_subaps, tSamples, n_axis)
        # LSTM output is (batch, tSamples, hidden_size), final hidden state (discarded), final cell state (discarded)
        out, _ = self.lstm(x_sorted)
        
        # We pass the last hidden state into the linear layer to produce the prediction
        # y is of size (batch_size * n_subaps, n_axis)
        y = self.head(out[:, -1, :])

        if self.n_axis == 2:
            # (batch_size * n_subaps, 2) -> (batch_size, n_subaps, 2)
            # -> permute to (batch_size, 2, n_subaps) -> (batch_size, nSlopes)
            y_out = (
                y.view(batch_size, n_subaps, self.n_axis)
                .permute(0, 2, 1)
                .contiguous()
                .view(batch_size, nSlopes)
            )
        else:
            # (batch_size * nSlopes, 1) -> (batch_size, nSlopes)
            y_out = y.view(batch_size, nSlopes)

        # Return (nSlopes,) if unbatched, or (batch_size, nSlopes) if batched
        return y_out.squeeze(0) if not is_batched else y_out