import torch
import torch.nn as nn

class IndependentSlopeLSTM(nn.Module):
    def __init__(self, n_axis= 1, hidden_size=32, num_layers=1):
        """
        n_axis: int
            Number of axis of the slopes to be predicted at the same time. Expected 1 or 2 (X|Y or X-Y)
        hidden_size:int
            Number of internal states - memory length
        num_layers: int
            Number of LSTM sequencially
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
        # x is received as (time, nSlopes) being nSlopes sorted as [x, y]
        tSamples, nSlopes = x.shape
        # Make a copy to sort the data in the expected way for the LSTM
        batch_size = nSlopes // self.n_axis
        
        if self.n_axis == 2:
            # (tSamples, nSlopes) -> (tSamples, 2, nSlopes//2) -> (nSlopes//2, tSamples, 2)
            x_sorted = x.view(tSamples, self.n_axis, batch_size).permute(2, 0, 1).contiguous()
        else:
            # (tSamples, nSlopes) -> (nSlopes, tSamples) -> (nSlopes, tSamples, 1)
            x_sorted = x.t().unsqueeze(-1)

        # LSTM expects input data in the shape (batch, time, feature) = (nSlopes//n_axis, tSamples, n_axis)
        # LSTM output is (batch, tSamples, hidden_size), final hidden state (discarded), final cell state (discarded)
        out, _ = self.lstm(x_sorted)
        
        # We pass the last hidden state into the linear layer to produce the prediction
        # y is of size (batch, n_axis)
        y = self.head(out[:, -1, :])
        if self.n_axis == 2:
            # (batch, n_axis) -> (nSlopes//n_axis, n_axis) --> (nSlopes,)
            return y.t().contiguous().view(-1)
        else:
            # (batch, 1) --> (nSlopes, 1) --> (nSlopes,)
            return y.view(-1) 