import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

class SlopesDataset(Dataset):
    def __init__(self, data: list, past_horizon: int, pred_horizon: int = 2):
        """
        data : list
            May contain the data of several atmosphere cases: list[(nSamples, nSlopes)], len(list) = nCases,
            each element must be torch.Tensor
        past_horizon : int 
            Past horizon
        pred_horizon: int
            Prediction horizon
        """
        self.data = data
        self.past_horizon = past_horizon
        self.pred_horizon = pred_horizon
        # Compute number of independent samples that can be obtained from the data length, past horizon and prediction horizon
        self.len_per_case = []

        for i in range(len(self.data)):
            self.len_per_case.append(len(self.data[i]) - past_horizon - pred_horizon + 1)

        self.len_per_case = np.array(self.len_per_case) 
        
        self.cum_samples = np.cumsum(self.len_per_case)

        self.n_samples = int(self.len_per_case.sum())

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Figure out in which case is the idx and the local index within the case
        case = int(np.searchsorted(self.cum_samples, idx, side='right'))
        start = 0 if case == 0 else self.cum_samples[case - 1]
        local_index = idx - start        
        # Past data
        x = self.data[case][local_index:local_index+self.past_horizon]
        # Prediction truth value
        y_truth = self.data[case][local_index+self.past_horizon+self.pred_horizon-1]

        return x, y_truth