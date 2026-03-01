import pathlib
import torch
import h5py
import matplotlib.pyplot as plt

from wfePredictorSAOS.predictor.tcnPredictor import SHWFS_TCN
from wfePredictorSAOS.predictor.validationClass import eval_sequence_T8, mask_2H_to_2HW

device = "cuda" if torch.cuda.is_available() else "cpu"

## Load dataset

path_to_val_datasets = 'C:/Users/nicolas/Downloads/validationData'

validationData = pathlib.Path(path_to_val_datasets)

list_datasets_val = list(validationData.rglob('*.h5'))
list_datasets_val = [str(list_datasets_val[i]) for i in range(len(list_datasets_val))] 

dataset_list_val = []

for i in range(len(list_datasets_val)):
    with h5py.File(list_datasets_val[i], 'r') as f:
        dataset_list_val.append(torch.from_numpy(f['LightPath_0']['slopes_2D']['data'][:].squeeze()).float())

seq_real = dataset_list_val[0]
pupil_mask  = dataset_list_val[0].sum(axis=0) != 0    

nSubap = seq_real.shape[-1]  # W

model = SHWFS_TCN(n_subap=nSubap, emb=128, tcn_layers=4, tcn_kernel=3, dropout=0.3).to(device)
state = torch.load("C:/Users/nicolas/Documents/code/best_model_T8.pt", map_location="cpu")
model.load_state_dict(state)

m_model, m_base, e_model, e_base = eval_sequence_T8(
    model, seq_real, nSubap, horizon=2, mask2H_W=pupil_mask, device=device
)

print("Mean MSE model   :", m_model)
print("Mean MSE baseline:", m_base)
print("Improvement (%): ", 100.0 * (m_base - m_model) / max(m_base, 1e-12))


plt.plot(e_base, label="Baseline (persistence)")
plt.plot(e_model, label="Model (TCN)")
plt.legend()
plt.title("Frame-by-frame MSE (t+2)")
plt.xlabel("time index")
plt.ylabel("MSE")
plt.show()