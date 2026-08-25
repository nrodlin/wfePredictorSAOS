import os
import json
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from wfePredictorSAOS.predictor.slopesDataset import SlopesDataset
from wfePredictorSAOS.predictor.independentLSTMModel import IndependentSlopeLSTM

### Load h5 File

def load_h5(filename):
    with h5py.File(filename, 'r') as f:
        data = torch.from_numpy(f['LightPath_0']['slopes_1D']['data'][:].squeeze()).float()
    return data

@torch.no_grad()
def evaluate_dataset(model, data_loader, criterion, device):
    """
    Evaluates model and persistence baseline on a given dataloader.
    Returns dictionary with MSE, RMSE, persistence metrics, and % improvement.
    """
    model.eval()
    total_model_loss = 0.0
    total_pers_loss = 0.0
    n_samples = 0

    for x, y_truth in data_loader:
        x = x.to(device)
        y_truth = y_truth.to(device)

        # Model prediction
        y_pred = model(x)
        loss_model = criterion(y_pred, y_truth)

        # Persistence baseline: predict y_truth (t+pred_horizon) using the latest history frame x[:, -1, :]
        y_pers = x[:, -1, :]
        loss_pers = criterion(y_pers, y_truth)

        batch_len = len(x)
        total_model_loss += loss_model.item() * batch_len
        total_pers_loss += loss_pers.item() * batch_len
        n_samples += batch_len

    mse_model = total_model_loss / max(n_samples, 1)
    mse_pers = total_pers_loss / max(n_samples, 1)
    rmse_model = float(np.sqrt(mse_model))
    rmse_pers = float(np.sqrt(mse_pers))
    improvement_pct = ((mse_pers - mse_model) / max(mse_pers, 1e-12)) * 100.0

    return {
        "mse_model": mse_model,
        "rmse_model": rmse_model,
        "mse_persistence": mse_pers,
        "rmse_persistence": rmse_pers,
        "improvement_pct": improvement_pct,
        "n_samples": n_samples,
    }

# Launch the final training
def train_final_model(
    train_data,
    val_data,
    test_data,
    test_data_by_atm,
    device,
    past_horizon=24,
    hidden_size=16,
    num_layers=1,
    n_axis=1,
    learning_rate=3e-4,
    weight_decay=1e-4,
    max_epochs=200,
    batch_size=64,
    pred_horizon=2,
    early_stopping_patience=25,
    min_delta=1e-4,
    model_save_path="best_model_IndepLSTM.pt",
    results_save_path="test_results.json",
):
    pin_memory = (device == "cuda" or (isinstance(device, torch.device) and device.type == "cuda"))

    print(f"\n{'='*90}")
    print(f"Final Model Configuration:")
    print(f"  past_horizon={past_horizon} | hidden_size={hidden_size} | num_layers={num_layers} | n_axis={n_axis}")
    print(f"  lr={learning_rate:.1e} | weight_decay={weight_decay:.1e} | batch_size={batch_size} | pred_horizon={pred_horizon}")
    print(f"{'='*90}")

    # Create training and validation datasets (Test data is strictly isolated during training)
    train_dataset = SlopesDataset(train_data, past_horizon, pred_horizon=pred_horizon)
    val_dataset   = SlopesDataset(val_data, past_horizon, pred_horizon=pred_horizon)

    print(f"  Training samples  : {len(train_dataset):,} (from {len(train_data)} datasets, Atmospheres 1-4)")
    print(f"  Validation samples: {len(val_dataset):,} (from {len(val_data)} datasets, Atmosphere 5)")
    print(f"  Testing samples   : {len(SlopesDataset(test_data, past_horizon, pred_horizon=pred_horizon)):,} (from {len(test_data)} datasets, Atmospheres 6-8, strictly held-out)\n")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )

    # Create model, optimizer, and scheduler
    model = IndependentSlopeLSTM(n_axis=n_axis, hidden_size=hidden_size, num_layers=num_layers).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=max_epochs, eta_min=1e-6
    )

    # Loss criterion
    criterion = torch.nn.MSELoss()

    best_val_loss = np.inf
    best_train_loss = np.inf
    best_epoch = 0
    best_model_state = None
    epochs_without_improvement = 0

    print("Starting training with validation monitoring...")
    for epoch in range(max_epochs):
        # Training loop
        model.train()
        total_train_loss = 0.0
        n_train_samples = 0

        for x, y_truth in train_loader:
            x = x.to(device)
            y_truth = y_truth.to(device)

            optim.zero_grad(set_to_none=True)
            y_pred = model(x)
            loss = criterion(y_pred, y_truth)
            loss.backward()
            optim.step()

            batch_len = len(x)
            total_train_loss += loss.item() * batch_len
            n_train_samples += batch_len

        train_loss = total_train_loss / max(n_train_samples, 1)

        # Validation loop (strictly on val_loader, NOT test)
        val_metrics = evaluate_dataset(model, val_loader, criterion, device)
        val_loss = val_metrics["mse_model"]

        # Step scheduler
        scheduler.step()

        # Check early stopping and save best weights based on validation loss
        if val_loss < best_val_loss * (1 - min_delta):
            best_val_loss = val_loss
            best_train_loss = train_loss
            best_epoch = epoch + 1
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
            is_best = True
        else:
            epochs_without_improvement += 1
            is_best = False

        # Log results every 10 epochs or on epoch 1
        if (epoch + 1) == 1 or (epoch + 1) % 10 == 0:
            star = " *" if is_best else ""
            print(
                f"  Epoch {epoch + 1:3d}/{max_epochs} | "
                f"Train MSE: {train_loss:.5e} | "
                f"Val MSE: {val_loss:.5e} (Pers: {val_metrics['mse_persistence']:.5e}, Imprv: {val_metrics['improvement_pct']:+.2f}%){star} | "
                f"LR: {optim.param_groups[0]['lr']:.2e}"
            )

        if epochs_without_improvement >= early_stopping_patience:
            print(f"\n  [Early stopping at epoch {epoch + 1}]")
            break

    # Save the best model checkpoint
    if best_model_state is not None:
        torch.save(best_model_state, model_save_path)
        print(f"\n✓ Best model saved to: {model_save_path}")
        model.load_state_dict(best_model_state)

    print(f"\n{'='*90}")
    print(f"Training Complete:")
    print(f"  Best Epoch          : {best_epoch}")
    print(f"  Best Train Loss     : {best_train_loss:.6e} (RMSE: {np.sqrt(best_train_loss):.5f})")
    print(f"  Best Validation Loss: {best_val_loss:.6e} (RMSE: {np.sqrt(best_val_loss):.5f})")
    print(f"{'='*90}\n")

    # =========================================================================
    # FINAL EVALUATION ON STRICTLY HELD-OUT TEST DATA (Atmospheres 6 to 8)
    # =========================================================================
    print(f"{'='*90}")
    print("FINAL EVALUATION ON HELD-OUT TEST DATA (Atmospheres 6, 7, 8):")
    print(f"{'='*90}")

    test_dataset = SlopesDataset(test_data, past_horizon, pred_horizon=pred_horizon)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )

    test_metrics = evaluate_dataset(model, test_loader, criterion, device)

    print(f"\nGlobal Test Performance ({test_metrics['n_samples']:,} samples):")
    print(f"  Model MSE       : {test_metrics['mse_model']:.6e} (RMSE: {test_metrics['rmse_model']:.5f})")
    print(f"  Persistence MSE : {test_metrics['mse_persistence']:.6e} (RMSE: {test_metrics['rmse_persistence']:.5f})")
    print(f"  Improvement     : {test_metrics['improvement_pct']:+.2f}% over Persistence\n")

    # Per-Atmosphere Breakdown on Test Set
    per_atm_results = {}
    print(f"{'-'*90}")
    print(f"{'Atmosphere':<15} | {'Model MSE':<14} | {'Pers. MSE':<14} | {'Model RMSE':<12} | {'Pers. RMSE':<12} | {'Improvement':<12}")
    print(f"{'-'*90}")

    for atm_id, atm_files_data in sorted(test_data_by_atm.items()):
        atm_ds = SlopesDataset(atm_files_data, past_horizon, pred_horizon=pred_horizon)
        atm_loader = DataLoader(atm_ds, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
        m = evaluate_dataset(model, atm_loader, criterion, device)
        per_atm_results[f"Atmosphere_{atm_id}"] = m
        print(
            f"Atmosphere {atm_id:<4} | "
            f"{m['mse_model']:<14.6e} | "
            f"{m['mse_persistence']:<14.6e} | "
            f"{m['rmse_model']:<12.5f} | "
            f"{m['rmse_persistence']:<12.5f} | "
            f"{m['improvement_pct']:+10.2f}%"
        )
    print(f"{'-'*90}\n")

    # Save detailed evaluation to json
    results_dict = {
        "model_config": {
            "past_horizon": past_horizon,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "n_axis": n_axis,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "pred_horizon": pred_horizon,
            "batch_size": batch_size,
        },
        "training_summary": {
            "best_epoch": best_epoch,
            "best_train_mse": float(best_train_loss),
            "best_val_mse": float(best_val_loss),
        },
        "test_results_global": test_metrics,
        "test_results_per_atmosphere": per_atm_results,
    }

    with open(results_save_path, "w") as f:
        json.dump(results_dict, f, indent=4)
    print(f"✓ Test results saved to: {results_save_path}\n")

    return results_dict

# Entry point
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # Final hyperparameters selected by user
    past_horizon = 24
    hidden_size = 16
    num_layers = 1
    n_axis = 1
    learning_rate = 3e-4
    weight_decay = 1e-4
    max_epochs = 200
    batch_size = 64
    pred_horizon = 2
    model_save_path = "best_model_IndepLSTM.pt"
    results_save_path = "test_results.json"

    # Paths configuration
    dalia_path = '/net/dalia/scratch1/nlinares/results/results/predictor/training'
    local_path = '/home/nlinares/simulations/results/predictorSAOS/training'

    if os.path.exists(dalia_path):
        base_path = dalia_path
    elif os.path.exists(local_path):
        base_path = local_path
    else:
        base_path = dalia_path

    train_path = os.path.join(base_path, 'training_val')
    test_path = os.path.join(base_path, 'test')

    print(f"Training/Val data directory: {train_path}")
    print(f"Testing data directory     : {test_path}\n")

    # Filenames for Training (Atmospheres 1 to 4)
    filenames_train_by_atm = [
        # Atmosphere 1
        [
            f'{train_path}/res_atm1_draw1.h5',
            f'{train_path}/res_atm1_draw1_vib1.h5',
            f'{train_path}/res_atm1_draw2.h5',
            f'{train_path}/res_atm1_draw2_vib1.h5',
        ],
        # Atmosphere 2
        [
            f'{train_path}/res_atm2_draw1.h5',
            f'{train_path}/res_atm2_draw1_vib2.h5',
            f'{train_path}/res_atm2_draw2.h5',
            f'{train_path}/res_atm2_draw2_vib2.h5',
        ],
        # Atmosphere 3
        [
            f'{train_path}/res_atm3_draw1.h5',
            f'{train_path}/res_atm3_draw1_vib3.h5',
            f'{train_path}/res_atm3_draw2.h5',
            f'{train_path}/res_atm3_draw2_vib3.h5',
        ],
        # Atmosphere 4
        [
            f'{train_path}/res_atm4_draw1.h5',
            f'{train_path}/res_atm4_draw1_vib4.h5',
            f'{train_path}/res_atm4_draw2.h5',
            f'{train_path}/res_atm4_draw2_vib4.h5',
        ],
    ]

    # Filenames for Validation during training (Atmosphere 5)
    filenames_val_by_atm = [
        # Atmosphere 5
        [
            f'{train_path}/res_atm5_draw1.h5',
            f'{train_path}/res_atm5_draw1_vib5.h5',
            f'{train_path}/res_atm5_draw2.h5',
            f'{train_path}/res_atm5_draw2_vib5.h5',
        ],
    ]

    # Filenames for Testing (Atmospheres 6 to 8) - strictly held-out
    filenames_test_by_atm = {
        6: [
            f'{test_path}/res_atm6_draw1.h5',
            f'{test_path}/res_atm6_draw1_vib6.h5',
            f'{test_path}/res_atm6_draw2.h5',
            f'{test_path}/res_atm6_draw2_vib6.h5',
        ],
        7: [
            f'{test_path}/res_atm7_draw1.h5',
            f'{test_path}/res_atm7_draw1_vib7.h5',
            f'{test_path}/res_atm7_draw2.h5',
            f'{test_path}/res_atm7_draw2_vib7.h5',
        ],
        8: [
            f'{test_path}/res_atm8_draw1.h5',
            f'{test_path}/res_atm8_draw1_vib8.h5',
            f'{test_path}/res_atm8_draw2.h5',
            f'{test_path}/res_atm8_draw2_vib8.h5',
        ],
    }

    print("Loading Training datasets (Atmospheres 1 to 4)...")
    train_data = []
    for atm_idx, atm_files in enumerate(filenames_train_by_atm, 1):
        atm_count = 0
        for fpath in atm_files:
            if os.path.exists(fpath):
                train_data.append(load_h5(fpath))
                atm_count += 1
            else:
                print(f"  [WARNING] File not found: {fpath}")
        print(f"  Atmosphere {atm_idx}: loaded {atm_count} datasets")
    print(f"Total training datasets loaded: {len(train_data)}\n")

    print("Loading Validation datasets (Atmosphere 5)...")
    val_data = []
    for fpath in filenames_val_by_atm[0]:
        if os.path.exists(fpath):
            val_data.append(load_h5(fpath))
        else:
            print(f"  [WARNING] File not found: {fpath}")
    print(f"Total validation datasets loaded: {len(val_data)}\n")

    print("Loading Testing datasets (Atmospheres 6 to 8)...")
    test_data = []
    test_data_by_atm = {}
    for atm_idx, atm_files in filenames_test_by_atm.items():
        atm_count = 0
        atm_loaded = []
        for fpath in atm_files:
            if os.path.exists(fpath):
                d = load_h5(fpath)
                test_data.append(d)
                atm_loaded.append(d)
                atm_count += 1
            else:
                print(f"  [WARNING] File not found: {fpath}")
        test_data_by_atm[atm_idx] = atm_loaded
        print(f"  Atmosphere {atm_idx}: loaded {atm_count} datasets")
    print(f"Total testing datasets loaded: {len(test_data)}\n")

    # Launch the final training
    results = train_final_model(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        test_data_by_atm=test_data_by_atm,
        device=device,
        past_horizon=past_horizon,
        hidden_size=hidden_size,
        num_layers=num_layers,
        n_axis=n_axis,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        batch_size=batch_size,
        pred_horizon=pred_horizon,
        early_stopping_patience=25,
        min_delta=1e-4,
        model_save_path=model_save_path,
        results_save_path=results_save_path,
    )
