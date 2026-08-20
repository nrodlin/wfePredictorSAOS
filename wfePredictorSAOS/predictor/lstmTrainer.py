import os
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

# Launch the final training
def train_final_model(
    train_data,
    test_data,
    device,
    past_horizon=16,
    hidden_size=64,
    num_layers=2,
    n_axis=1,
    learning_rate=3e-4,
    weight_decay=1e-4,
    max_epochs=200,
    batch_size=256,
    pred_horizon=2,
    model_save_path="best_model_IndepLSTM.pt",
):
    pin_memory = (device == "cuda" or (isinstance(device, torch.device) and device.type == "cuda"))

    print(f"\n{'='*80}")
    print(f"Final Model Configuration: past_horizon={past_horizon} | hidden_size={hidden_size} | "
          f"num_layers={num_layers} | n_axis={n_axis} | lr={learning_rate:.1e} | weight_decay={weight_decay:.1e}")
    print(f"{'='*80}")

    # Create datasets
    train_dataset = SlopesDataset(train_data, past_horizon, pred_horizon=pred_horizon)
    test_dataset  = SlopesDataset(test_data, past_horizon, pred_horizon=pred_horizon)

    print(f"  Training samples: {len(train_dataset):,} (from {len(train_data)} datasets, Atmospheres 1-5)")
    print(f"  Testing samples : {len(test_dataset):,} (from {len(test_data)} datasets, Atmospheres 6-8)\n")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
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

    early_stopping_patience = 25
    min_delta = 1e-4

    best_test_loss = np.inf
    best_train_loss = np.inf
    best_epoch = 0
    best_model_state = None
    epochs_without_improvement = 0

    print("Starting training...")
    for epoch in range(max_epochs):
        # Training
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

        # Validation on test set
        model.eval()
        total_test_loss = 0.0
        n_test_samples = 0

        with torch.no_grad():
            for x, y_truth in test_loader:
                x = x.to(device)
                y_truth = y_truth.to(device)

                y_pred = model(x)
                loss = criterion(y_pred, y_truth)

                batch_len = len(x)
                total_test_loss += loss.item() * batch_len
                n_test_samples += batch_len

        test_loss = total_test_loss / max(n_test_samples, 1)

        # Step scheduler
        scheduler.step()

        # Check early stopping and save best weights
        if test_loss < best_test_loss * (1 - min_delta):
            best_test_loss = test_loss
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
                f"Train MSE: {train_loss:.5e} (RMSE: {np.sqrt(train_loss):.4f}) | "
                f"Test MSE: {test_loss:.5e} (RMSE: {np.sqrt(test_loss):.4f}){star} | "
                f"LR: {optim.param_groups[0]['lr']:.2e}"
            )

        if epochs_without_improvement >= early_stopping_patience:
            print(f"\n  [Early stopping at epoch {epoch + 1}]")
            break

    # Save the best model checkpoint
    if best_model_state is not None:
        torch.save(best_model_state, model_save_path)
        print(f"\n✓ Best model saved to: {model_save_path}")

    print(f"\n{'='*80}")
    print(f"Training Complete:")
    print(f"  Best Epoch     : {best_epoch}")
    print(f"  Best Train Loss: {best_train_loss:.6e} (RMSE: {np.sqrt(best_train_loss):.5f})")
    print(f"  Best Test Loss : {best_test_loss:.6e} (RMSE: {np.sqrt(best_test_loss):.5f})")
    print(f"{'='*80}\n")

    return {
        "best_epoch": best_epoch,
        "best_train_loss": best_train_loss,
        "best_test_loss": best_test_loss,
        "model_save_path": model_save_path,
    }

# Entry point
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # Final hyperparameters (set according to k-fold selection)
    past_horizon = 16
    hidden_size = 64
    num_layers = 2
    n_axis = 1
    learning_rate = 3e-4
    weight_decay = 1e-4
    max_epochs = 200
    batch_size = 8
    pred_horizon = 2
    model_save_path = "best_model_IndepLSTM.pt"

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

    print(f"Training data directory: {train_path}")
    print(f"Testing data directory : {test_path}\n")

    # Read filenames for Training (Atmospheres 1 to 5)
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
        # Atmosphere 5
        [
            f'{train_path}/res_atm5_draw1.h5',
            f'{train_path}/res_atm5_draw1_vib5.h5',
            f'{train_path}/res_atm5_draw2.h5',
            f'{train_path}/res_atm5_draw2_vib5.h5',
        ],
    ]

    # Read filenames for Testing (Atmospheres 6 to 8)
    filenames_test_by_atm = [
        # Atmosphere 6
        [
            f'{test_path}/res_atm6_draw1.h5',
            f'{test_path}/res_atm6_draw1_vib6.h5',
            f'{test_path}/res_atm6_draw2.h5',
            f'{test_path}/res_atm6_draw2_vib6.h5',
        ],
        # Atmosphere 7
        [
            f'{test_path}/res_atm7_draw1.h5',
            f'{test_path}/res_atm7_draw1_vib7.h5',
            f'{test_path}/res_atm7_draw2.h5',
            f'{test_path}/res_atm7_draw2_vib7.h5',
        ],
        # Atmosphere 8
        [
            f'{test_path}/res_atm8_draw1.h5',
            f'{test_path}/res_atm8_draw1_vib8.h5',
            f'{test_path}/res_atm8_draw2.h5',
            f'{test_path}/res_atm8_draw2_vib8.h5',
        ],
    ]

    print("Loading Training datasets (Atmospheres 1 to 5)...")
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

    print("Loading Testing datasets (Atmospheres 6 to 8)...")
    test_data = []
    for atm_idx, atm_files in enumerate(filenames_test_by_atm, 6):
        atm_count = 0
        for fpath in atm_files:
            if os.path.exists(fpath):
                test_data.append(load_h5(fpath))
                atm_count += 1
            else:
                print(f"  [WARNING] File not found: {fpath}")
        print(f"  Atmosphere {atm_idx}: loaded {atm_count} datasets")
    print(f"Total testing datasets loaded: {len(test_data)}\n")

    # Launch the final training
    results = train_final_model(
        train_data=train_data,
        test_data=test_data,
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
        model_save_path=model_save_path,
    )
