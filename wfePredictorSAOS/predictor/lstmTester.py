import os
import pathlib
import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import json

from wfePredictorSAOS.predictor.slopesDataset import SlopesDataset
from wfePredictorSAOS.predictor.independentLSTMModel import IndependentSlopeLSTM

# Reuse normalization code logic, keeping comments in English
def apply_normalization_per_file(dataset_list, label=""):
    print(f"Normalizing {label} per file:")
    normalized, stats = [], []
    for i, d in enumerate(dataset_list):
        m = d.mean(dim=0)
        s = d.std(dim=0).clamp(min=1e-6)
        normalized.append((d - m) / s)
        stats.append((m, s))
        print(f"  file {i:02d}: mean={m.mean():.4f}  std={s.mean():.4f}  "
              f"frames={d.shape[0]}")
    print()
    return normalized, stats

# Porting all the beautiful visualization plots required in ML
@torch.no_grad()
def collect_predictions(model, test_ds, device, n_samples=5):
    """
    Collects a fixed number of samples from the test set randomly.
      - xs      : (n_samples, past_horizon, nSlopes)  — past history 
      - ys_true : (n_samples, nSlopes)                — true future
      - ys_pred : (n_samples, nSlopes)                — model prediction
    """
    model.eval()
    indices = torch.randperm(len(test_ds))[:n_samples]
    xs, ys_true, ys_pred = [], [], []
    for i in indices:
        x, y = test_ds[i]
        xs.append(x)
        ys_true.append(y)
        pred = model(x.unsqueeze(0).to(device)).squeeze(0).cpu()
        ys_pred.append(pred)
    return (torch.stack(xs),
            torch.stack(ys_true),
            torch.stack(ys_pred))

def plot_slope_predictions(x_seq, y_true, y_pred, n_slopes_shown=8,
                           sample_idx=0, future_horizon=2):
    """
    Plots the specific trace of n_slopes_shown selected uniformly.
    - Past History (blue)
    - True target at t+future_horizon (green dot) 
    - Prediction (red cross)
    """
    past  = x_seq[sample_idx].numpy()    
    truth = y_true[sample_idx].numpy()   
    pred  = y_pred[sample_idx].numpy()   

    T = past.shape[0]
    t_past = np.arange(T)
    t_pred = T + future_horizon - 1      

    # Choose slopes
    slope_indices = np.linspace(0, past.shape[1] - 1, n_slopes_shown, dtype=int)

    ncols = 4
    nrows = int(np.ceil(n_slopes_shown / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4, nrows * 2.8),
                             sharey=False)
    axes = axes.flatten()

    for plot_i, s in enumerate(slope_indices):
        ax = axes[plot_i]
        ax.plot(t_past, past[:, s], color="steelblue",
                linewidth=1.5, marker="o", markersize=3, label="History")
        ax.scatter(t_pred, truth[s], color="green",
                   s=80, zorder=5, label="Truth")
        ax.scatter(t_pred, pred[s],  color="red",
                   s=80, marker="x", zorder=5, linewidths=2, label="Prediction")
        error = abs(truth[s] - pred[s])
        ax.set_title(f"Slope {s}  |  |err|={error:.4f}", fontsize=9)
        ax.axvline(T - 1, color="gray", linestyle=":", linewidth=0.8)
        ax.set_xlabel("Frame (normalized)")
        if plot_i == 0:
            ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for j in range(plot_i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Prediction per slope — test sample #{sample_idx}", fontsize=12)
    plt.tight_layout()
    fname = f"test_predictions_per_slope_sample{sample_idx}.png"
    plt.savefig(fname, dpi=150)
    print(f"[PLOT] {fname} saved")

def plot_full_vector_comparison(y_true, y_pred, nSlopes, n_samples=3):
    """
    Compares the complete vector of Slopes spanning the WFS, identifying
    any continuous systematic error fields.
    """
    fig, axes = plt.subplots(n_samples, 1,
                             figsize=(14, 3.5 * n_samples),
                             sharex=True)
    if n_samples == 1:
        axes = [axes]

    slope_axis = np.arange(nSlopes)
    half = nSlopes // 2  

    for i, ax in enumerate(axes):
        truth = y_true[i].numpy()
        pred  = y_pred[i].numpy()
        error = truth - pred

        ax.plot(slope_axis, truth, color="green",    linewidth=0.8, alpha=0.9, label="Truth")
        ax.plot(slope_axis, pred,  color="red",      linewidth=0.8, alpha=0.9, label="Prediction")
        ax.fill_between(slope_axis, 0, error,        color="orange",alpha=0.4, label="Error")
        ax.axvline(half, color="gray", linestyle="--", linewidth=1, label="X | Y")
        ax.set_ylabel("Amplitude (norm.)")
        ax.set_title(f"Test sample #{i}  —  MSE={np.mean(error**2):.6f}", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.2)
        
    axes[-1].set_xlabel("Slope Index (left: X, right: Y)")
    fig.suptitle("Full Slope Vector: Truth vs Prediction", fontsize=13)
    plt.tight_layout()
    plt.savefig("test_full_vector_comparison.png", dpi=150)
    print("[PLOT] test_full_vector_comparison.png saved")

def plot_error_distribution(y_true, y_pred):
    """
    Histogram of overall prediction error, checking for bias shifts.
    """
    error = (y_true - y_pred).numpy().flatten()
    mse   = np.mean(error ** 2)
    mae   = np.mean(np.abs(error))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(error, bins=80, color="steelblue", edgecolor="none", alpha=0.8)
    axes[0].axvline(0, color="red", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Error (Pred - Truth)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title(f"Error Distribution\nMSE={mse:.6f}  MAE={mae:.6f}")
    axes[0].grid(True, alpha=0.3)

    error_2d     = (y_true - y_pred).numpy()         
    mean_abs_err = np.abs(error_2d).mean(axis=0)     
    axes[1].plot(mean_abs_err, linewidth=0.8, color="darkorange")
    axes[1].axvline(mean_abs_err.shape[0] // 2, color="gray",
                    linestyle="--", linewidth=1, label="X | Y")
    axes[1].set_xlabel("Slope Index")
    axes[1].set_ylabel("Mean |Error|")
    axes[1].set_title("Mean Absolute Error per Slope")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("test_error_distribution.png", dpi=150)
    print("[PLOT] test_error_distribution.png saved")

def plot_temporal_sequence(model, test_data_raw, device, past_horizon,
                           future_horizon, n_frames=80, slope_indices=None):
    """
    Plots a continuous loop trace comparing the physical timeseries tracking dynamics.
    """
    if slope_indices is None:
        nS = test_data_raw.shape[1]
        slope_indices = np.linspace(0, nS - 1, 6, dtype=int)

    model.eval()
    data = test_data_raw  

    preds, truths = [], []
    with torch.no_grad():
        for t in range(n_frames):
            start = t
            end   = t + past_horizon
            if end + future_horizon > len(data):
                break
            x    = data[start:end].unsqueeze(0).to(device)   
            pred = model(x).squeeze(0).cpu()
            truth = data[end + future_horizon - 1]
            preds.append(pred)
            truths.append(truth)

    preds  = torch.stack(preds).numpy()   
    truths = torch.stack(truths).numpy()  
    t_axis = np.arange(len(preds))

    ncols = 3
    nrows = int(np.ceil(len(slope_indices) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 5, nrows * 3),
                             sharey=False)
    axes = axes.flatten()

    for plot_i, s in enumerate(slope_indices):
        ax = axes[plot_i]
        ax.plot(t_axis, truths[:, s], color="green", linewidth=1.2, label="Truth", alpha=0.9)
        ax.plot(t_axis, preds[:, s],  color="red", linewidth=1.0, label="Prediction", alpha=0.9, linestyle="--")
        corr = np.corrcoef(truths[:, s], preds[:, s])[0, 1]
        ax.set_title(f"Slope {s}  |  r={corr:.3f}", fontsize=9)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Amplitude (norm.)")
        if plot_i == 0:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for j in range(plot_i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Temporal Evolution Dynamics: Truth vs Prediction (Test file 0)", fontsize=12)
    plt.tight_layout()
    plt.savefig("test_temporal_sequence.png", dpi=150)
    print("[PLOT] test_temporal_sequence.png saved")

# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # ── Paths ─────────────────────────────────────────────────────────────────
    user_home = os.path.expanduser('~')
    results_path = os.path.join(user_home, "simulations", "results", "predictorSAOS", "training", "test")
    model_path = "best_model_IndepLSTM.pt"
    
    # ── Hyperparameters ───────────────────────────────────────────────────────
    past_horizon   = 8
    future_horizon = 2
    stride         = 1
    hidden_size    = 32

    # ── Verify the Model ──────────────────────────────────────────────────────
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found. Ensure you run lstmTrainer.py first.")
        
    # ── Load Model ────────────────────────────────────────────────────────────
    model = IndependentSlopeLSTM(hidden_size=hidden_size, num_layers=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # ── Load Testing Data (Cases 6 to 8) ──────────────────────────────────────
    list_datasets_test = []
    for case_num in range(6, 9):
        for draw_num in range(1, 3):
            file_names = [
                f"res_atm{case_num}_draw{draw_num}.h5",
                f"res_atm{case_num}_draw{draw_num}_vib{case_num}.h5"
            ]
            for file_name in file_names:
                file_path = os.path.join(results_path, file_name)
                if os.path.exists(file_path):
                    list_datasets_test.append(file_path)
                else:
                    print(f"[WARNING] Expected test file missing: {file_path}")
                
    if not list_datasets_test:
        raise ValueError("No testing data files found.")

    dataset_list_test = []
    for fp in list_datasets_test:
        with h5py.File(fp, 'r') as f:
            dataset_list_test.append(
                torch.from_numpy(
                    f['LightPath_0']['slopes_1D']['data'][:].squeeze()
                ).float()
            )

    nSlopes = dataset_list_test[0].shape[-1]
    print(f"nSlopes = {nSlopes}\n")
    
    # ── Normalization ─────────────────────────────────────────────────────────
    # Keeping per-file normalization rule
    dataset_list_test, _ = apply_normalization_per_file(dataset_list_test, "test")
    
    # Assemble test dataloader
    test_ds = ConcatDataset([
        SlopesDataset(ds, past_horizon, future_horizon, stride)
        for ds in dataset_list_test
    ])
    
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)

    # ── System Testing Metrics ────────────────────────────────────────────────
    print("Initiating full test evaluation...")
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = F.mse_loss(pred, y)
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
            
    final_test_loss = total_loss / max(n, 1)
    
    print(f"\n[GLOBAL] Test Set (Cases 6-8) Total Accumulated MSE = {final_test_loss:.6f}")
    
    # Export metrics json
    with open("test_results.json", "w") as f:
        json.dump({
            "num_test_samples": len(test_ds),
            "mse_loss": final_test_loss
        }, f, indent=4)
        
    print("[EXPORT] test_results.json saved")
    
    # ── Test Set Plotting Extracted from Old Trainer ──────────────────────────
    
    x_seq, y_true, y_pred = collect_predictions(model, test_ds, device, n_samples=5)

    plot_slope_predictions(x_seq, y_true, y_pred, n_slopes_shown=8, sample_idx=0, future_horizon=future_horizon)
    plot_full_vector_comparison(y_true, y_pred, nSlopes, n_samples=3)

    x_all, y_true_all, y_pred_all = collect_predictions(
        model, test_ds, device, n_samples=min(200, len(test_ds))
    )
    plot_error_distribution(y_true_all, y_pred_all)

    plot_temporal_sequence(
        model,
        test_data_raw=dataset_list_test[0],  
        device=device,
        past_horizon=past_horizon,
        future_horizon=future_horizon,
        n_frames=150,
        slope_indices=None,                  
    )
