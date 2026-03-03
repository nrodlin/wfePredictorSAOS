import pathlib
import torch
import h5py
import matplotlib.pyplot as plt
import random
from wfePredictorSAOS.predictor.tcnPredictor import SHWFS_TCN, masked_mse, eval_model_and_baseline

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Helpers
# -----------------------------
@torch.no_grad()
def predict_one(model, window_2H_W, nSubap, device, mask2=None):
    """
    window_2H_W: (T, 2H, W)
    returns pred: (2,H,W)
    """
    H = nSubap
    xX = window_2H_W[:, :H, :]
    xY = window_2H_W[:, H:, :]
    x = torch.stack([xX, xY], dim=1)  # (T,2,H,W)
    x = x.unsqueeze(0).to(device)     # (1,T,2,H,W)
    pred = model(x).squeeze(0)        # (2,H,W)
    if mask2 is not None:
        pred = pred * mask2.to(pred.device)
    return pred

@torch.no_grad()
def eval_sequence(model, seq_real, nSubap, T=8, horizon=2, mask2=None, device="cuda"):
    """
    seq_real: (N, 2H, W)
    mask2: (2,H,W) float {0,1}
    returns:
      m_model, m_base, e_model, e_base
    """
    model.eval()
    seq = seq_real.to(device)
    H = nSubap
    N = seq.shape[0]

    e_model = []
    e_base = []

    for t in range(T-1, N-horizon):
        window = seq[t-T+1:t+1]           # (T,2H,W)
        target2H = seq[t+horizon]         # (2H,W)

        # target (2,H,W)
        target = torch.stack([target2H[:H, :], target2H[H:, :]], dim=0)
        if mask2 is not None:
            target = target * mask2.to(device)

        # baseline persistence: frame t -> predict t+h
        base2H = seq[t]
        base = torch.stack([base2H[:H, :], base2H[H:, :]], dim=0)
        if mask2 is not None:
            base = base * mask2.to(device)

        # model
        pred = predict_one(model, window, nSubap, device=device, mask2=mask2)

        if mask2 is None:
            em = torch.mean((pred - target) ** 2).item()
            eb = torch.mean((base - target) ** 2).item()
        else:
            em = masked_mse(pred.unsqueeze(0), target.unsqueeze(0), mask2).item()
            eb = masked_mse(base.unsqueeze(0), target.unsqueeze(0), mask2).item()

        e_model.append(em)
        e_base.append(eb)

    m_model = sum(e_model) / max(len(e_model), 1)
    m_base = sum(e_base) / max(len(e_base), 1)
    return m_model, m_base, e_model, e_base


# -----------------------------
# Load validation dataset
# -----------------------------
path_to_val_datasets = "C:/Users/nicolas/Downloads/validationData"
validationData = pathlib.Path(path_to_val_datasets)
list_datasets_val = [str(p) for p in validationData.rglob("*.h5")]

dataset_list_val = []
for fp in list_datasets_val:
    with h5py.File(fp, "r") as f:
        dataset_list_val.append(torch.from_numpy(
            f["LightPath_0"]["slopes_2D"]["data"][:].squeeze()
        ).float())

seq_real = dataset_list_val[0]                    # (N,2H,W)
nSubap = seq_real.shape[-1]
H = nSubap

# mask2 (2,H,W)
pupil_mask2H = (seq_real.sum(dim=0) != 0)         # bool (2H,W)
mask2 = torch.stack([pupil_mask2H[:H, :], pupil_mask2H[H:, :]], dim=0).float()

# -----------------------------
# Load model
# -----------------------------
model = SHWFS_TCN(n_subap=nSubap, emb=128, tcn_layers=4, tcn_kernel=3, dropout=0.3).to(device)
state = torch.load("C:/Users/nicolas/Documents/code/wfePredictorSAOS/best_model_T8.pt", map_location="cpu")
model.load_state_dict(state)
model.eval()

# -----------------------------
# Evaluate full sequence
# -----------------------------
T = 8
horizon = 2
m_model, m_base, e_model, e_base = eval_model_and_baseline(
    model, seq_real, nSubap, T=8, horizon=2, mask2=mask2, device=device
)

print("Mean MSE model   :", m_model)
print("Mean MSE baseline:", m_base)
print("Improvement (%):", 100.0 * (m_base - m_model) / max(m_base, 1e-12))

plt.figure()
plt.plot(e_base, label="Baseline (persistence)")
plt.plot(e_model, label="Model (TCN)")
plt.legend()
plt.title(f"Frame-by-frame masked MSE (t+{horizon}) | T={T}")
plt.xlabel("time index")
plt.ylabel("MSE")
plt.show()

# -----------------------------
# Visualización en subplots 4x3
# -----------------------------
N = seq_real.shape[0]
T = 8
horizon = 2

valid_times = list(range(T-1, N-horizon))
random_times = random.sample(valid_times, 4)

for channel, ch_name in zip([0, 1], ["X", "Y"]):

    fig, axes = plt.subplots(4, 3, figsize=(12, 12))
    fig.suptitle(f"Channel {ch_name} | Prediction vs Real vs Error", fontsize=14)

    for row, t in enumerate(random_times):

        window = seq_real[t-T+1:t+1]
        pred = predict_one(model, window, nSubap, device=device, mask2=mask2).cpu()

        target2H = seq_real[t+horizon]
        target = torch.stack([target2H[:H, :], target2H[H:, :]], dim=0)
        target = (target * mask2).cpu()

        error = pred - target

        p = pred[channel].numpy()
        r = target[channel].numpy()
        e = error[channel].numpy()

        vmax = max(abs(p).max(), abs(r).max())
        emax = abs(e).max()

        axes[row, 0].imshow(p, vmin=-vmax, vmax=vmax)
        axes[row, 0].set_title(f"Pred (t={t})")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(r, vmin=-vmax, vmax=vmax)
        axes[row, 1].set_title("Real")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(e, vmin=-emax, vmax=emax)
        axes[row, 2].set_title("Error")
        axes[row, 2].axis("off")

    plt.tight_layout()
    plt.show()

    print('end')