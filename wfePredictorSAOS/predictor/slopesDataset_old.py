import torch

class SlopesDataset(torch.utils.data.Dataset):
    def __init__(self, data, past_horizon, prediction_horizon, stride=1):
        """
        data : torch.Tensor
            Data in format nFrames, nSlopes
        past_horizon : int
            Discrete samples to look into the past data
        prediction_horizon : int
            Discrete sample to predict into the future
        stride : int, optional
            Discrete sample that are ignored to avoid using similar data. Default is 1.
        """
        super().__init__()

        self.data = data
        self.past_horizon = past_horizon
        self.prediction_horizon = prediction_horizon
        self.stride = stride

        if self.stride < 1:
            raise ValueError("stride must be >= 1")

        self.initial_index = []

        last_start = data.shape[0] - past_horizon - prediction_horizon + 1
        for i in range(0, last_start, self.stride):
            self.initial_index.append(i)

    def __len__(self):
        return len(self.initial_index)

    def __getitem__(self, k):
        index = self.initial_index[k]
        past_data = self.data[index:index + self.past_horizon]
        prediction_result = self.data[index + self.past_horizon + self.prediction_horizon - 1]

        return past_data, prediction_result