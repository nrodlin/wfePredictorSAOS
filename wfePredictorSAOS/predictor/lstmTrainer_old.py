import pathlib
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import json

from wfePredictorSAOS.predictor.slopesDataset import SlopesDataset
from wfePredictorSAOS.predictor.independentLSTMModel import IndependentSlopeLSTM

# ==============================================================================
# NORMALIZATION
# ==============================================================================
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

# ==============================================================================
# DATASETS
# ==============================================================================
def make_datasets(train_data, val_data, past_horizon, future_horizon, stride):
    train_ds = ConcatDataset([
        SlopesDataset(ds, past_horizon, future_horizon, stride)
        for ds in train_data
    ])
    val_ds = ConcatDataset([
        SlopesDataset(ds, past_horizon, future_horizon, stride)
        for ds in val_data
    ])
    return train_ds, val_ds

# ==============================================================================
# MODEL, OPTIMIZER AND SCHEDULER
# ==============================================================================
def build_model(hidden_size, lr, weight_decay, device):
    model = IndependentSlopeLSTM(hidden_size=hidden_size, num_layers=1).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] Trainable parameters: {n_params:,}\n")
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", factor=0.5, patience=8,
        threshold=1e-4, threshold_mode="rel", min_lr=1e-7,
    )
    return model, optim, scheduler

# ==============================================================================
# TRAIN / EVAL
# ==============================================================================
def train_one_epoch(model, loader, optim, device):
    model.train()
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = F.mse_loss(pred, y)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)

@torch.no_grad()
def eval_one_epoch(model, loader, device):
    model.eval()
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = F.mse_loss(pred, y)
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)

# ==============================================================================
# MAIN TRAINING LOOP (K-FOLD)
# ==============================================================================
def train_kfold(k_folds, dataset_list, baseline_mse, hidden_size, lr, patience, 
                weight_decay, n_epochs, past_horizon, future_horizon, stride, device):
    
    # Track the global absolute best model across all folds
    global_best_val = float("inf")
    global_best_state = None
    global_best_fold = -1
    global_best_epoch = -1
    
    # Saving histories
    history = {"folds": []}
    
    # Data is split mechanically since it has 10 elements: E.g., for fold i, files corresponding to the validation are [i * chunk_size:(i+1)*chunk_size].
    chunk_size = len(dataset_list) // k_folds
    if chunk_size == 0:
        chunk_size = 1
        k_folds = len(dataset_list)

    for fold in range(k_folds):
        print(f"\n{'='*50}")
        print(f"FOLD {fold + 1}/{k_folds}")
        print(f"{'='*50}")
        
        # Split datasets
        val_idx = list(range(fold * chunk_size, min((fold + 1) * chunk_size, len(dataset_list))))
        train_idx = [i for i in range(len(dataset_list)) if i not in val_idx]
        
        train_data = [dataset_list[i] for i in train_idx]
        val_data = [dataset_list[i] for i in val_idx]
        
        train_ds, val_ds = make_datasets(train_data, val_data, past_horizon, future_horizon, stride)
        
        model, optim, scheduler = build_model(hidden_size, lr, weight_decay, device)
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,
                                  num_workers=2, pin_memory=True, persistent_workers=True)
        val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False,
                                  num_workers=2, pin_memory=True, persistent_workers=True)
                                  
        print(f"==== Train samples: {len(train_ds)} | Val samples: {len(val_ds)} ====")
        print(f"     Baseline MSE to beat: {baseline_mse:.6f}\n")

        best_val, best_epoch = float("inf"), 0
        epochs_no_improve = 0
        train_losses, val_losses = [], []
        
        for epoch in range(1, n_epochs + 1):
            loss_train = train_one_epoch(model, train_loader, optim, device)
            loss_eval  = eval_one_epoch(model, val_loader, device)
            scheduler.step(loss_eval)
            current_lr = optim.param_groups[0]["lr"]
            
            train_losses.append(loss_train)
            val_losses.append(loss_eval)
            
            if loss_eval < best_val - 1e-4:
                best_val, best_epoch, epochs_no_improve = loss_eval, epoch, 0
                
                # Update global best model if it qualifies
                if best_val < global_best_val:
                    global_best_val = best_val
                    global_best_fold = fold + 1
                    global_best_epoch = epoch
                    global_best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    torch.save(global_best_state, "best_model_IndepLSTM.pt")
                    print("  --> [NEW GLOBAL BEST MODEL SAVED]")
            else:
                epochs_no_improve += 1

            beating = "✅" if loss_eval < baseline_mse * 0.95 else "⚠️ "
            print(f"fold={fold+1} epoch={epoch:03d}/{n_epochs} "
                  f"train={loss_train:.6f} val={loss_eval:.6f} "
                  f"best_val={best_val:.6f} {beating} "
                  f"lr={current_lr:.2e} no_improve={epochs_no_improve}/{patience}")

            if epochs_no_improve >= patience:
                print(f"\nEarly stopping on fold {fold+1}, epoch {epoch}. "
                      f"Best epoch={best_epoch}, best val={best_val:.6f}")
                break
                
        # Store fold history
        history["folds"].append({
            "fold": fold + 1,
            "train_loss": train_losses,
            "val_loss": val_losses,
            "best_epoch": best_epoch,
            "best_val": best_val
        })

    print(f"\n[K-FOLD FINISHED] Absolute best model was from Fold {global_best_fold}, Epoch {global_best_epoch} with Val Loss: {global_best_val:.6f}")
    
    # Save the training error evolution tracking
    with open("training_history.json", "w") as f:
        json.dump(history, f, indent=4)
        
    # Reload best state before returning just to be clean, though we won't strictly use it here anymore since tests are separated
    if global_best_state is not None:
        model.load_state_dict(global_best_state)
        
    return history


# ==============================================================================
# VISUALIZATION
# ==============================================================================
def plot_training_curve(history):
    fig, ax = plt.subplots(figsize=(10, 5))
    
    for fold_data in history["folds"]:
        fold_idx = fold_data["fold"]
        train_losses = fold_data["train_loss"]
        val_losses = fold_data["val_loss"]
        best_epoch = fold_data["best_epoch"]
        
        epochs = range(1, len(train_losses) + 1)
        
        # Plot train and val curves per fold
        line, = ax.plot(epochs, train_losses, label=f"Fold {fold_idx} Train", linewidth=1.2, linestyle='dotted')
        color = line.get_color()
        ax.plot(epochs, val_losses, label=f"Fold {fold_idx} Val", linewidth=1.5, color=color)
        ax.plot(best_epoch, val_losses[best_epoch - 1], marker='o', markersize=6, color=color)
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training Curve across Folds")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("kfold_training_curve.png", dpi=150)
    print("\n[PLOT] kfold_training_curve.png saved")

# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # ── Paths ─────────────────────────────────────────────────────────────────
    user_home = os.path.expanduser('~')
    results_path = os.path.join(user_home, "simulations", "results")

    # ── Hyperparameters ───────────────────────────────────────────────────────
    past_horizon   = 8
    future_horizon = 2
    stride         = 1
    n_epochs       = 250
    lr             = 1e-4
    weight_decay   = 1e-2
    patience       = 10
    hidden_size    = 32
    k_folds        = 5 # Divides the 10 loaded train datasets neatly into 5 chunks of 2 for val (80/20 train/val split per fold)

    # ── Data Loading ──────────────────────────────────────────────────────────
    # ONLY load training files (Cases 1 to 5)
    list_datasets_train = []
    
    # Append both draws for cases 1 to 5
    for case_num in range(1, 6):
        for draw_num in range(1, 3):
            file_names = [
                f"res_atm{case_num}_draw{draw_num}.h5",
                f"res_atm{case_num}_draw{draw_num}_vib{case_num}.h5"
            ]
            for file_name in file_names:
                file_path = os.path.join(results_path, file_name)
                if os.path.exists(file_path):
                    list_datasets_train.append(file_path)
                else:
                    print(f"[WARNING] Unfound expected train file: {file_path}")

    if not list_datasets_train:
        raise ValueError("No training files found in results dict.")

    dataset_list_train = []
    for fp in list_datasets_train:
        with h5py.File(fp, 'r') as f:
            dataset_list_train.append(
                torch.from_numpy(
                    f['LightPath_0']['slopes_1D']['data'][:].squeeze()
                ).float()
            )

    nSlopes = dataset_list_train[0].shape[-1]
    print(f"nSlopes = {nSlopes}\n")

    # ── Normalization ─────────────────────────────────────────────────────────
    dataset_list_train, _ = apply_normalization_per_file(dataset_list_train, "train")

    # ── Baseline ──────────────────────────────────────────────────────────────
    all_train    = torch.cat(dataset_list_train, dim=0)
    baseline_mse = all_train.var(dim=0).mean().item()
    print(f"[CHECK] Baseline MSE: {baseline_mse:.6f}\n")

    # ── K-Fold Training ───────────────────────────────────────────────────────
    history = train_kfold(
        k_folds, dataset_list_train, baseline_mse,
        hidden_size, lr, patience, weight_decay, n_epochs,
        past_horizon, future_horizon, stride, device
    )

    # ── Visualizations ────────────────────────────────────────────────────────
    plot_training_curve(history)