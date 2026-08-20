import os
import h5py
import numpy as np
import torch
import itertools

from torch.utils.data import DataLoader

from wfePredictorSAOS.predictor.slopesDataset import SlopesDataset
from wfePredictorSAOS.predictor.independentLSTMModel import IndependentSlopeLSTM

### Load h5 File

def load_h5(filename):
    with h5py.File(filename, 'r') as f:
        data = torch.from_numpy(f['LightPath_0']['slopes_1D']['data'][:].squeeze()).float()
    return data

# launch the training
def train_per_fold(data, combinations, device, max_epochs=200, batch_size=64, pred_horizon=2, weight_decay=1e-4):

    n_folds = len(data)
    pin_memory = (device == "cuda" or (isinstance(device, torch.device) and device.type == "cuda"))

    results = []

    n_combs = len(combinations)

    # Iterate with the iterator
    for comb_idx, combination in enumerate(combinations, 1):
        # Extract each hyperparameter from the combination
        past_horizon, hidden_size, num_layers, n_axis, learning_rate = combination

        print(f"\n{'='*80}")
        print(f"[{comb_idx:2d}/{n_combs:2d}] Configuration: past_horizon={past_horizon} | hidden_size={hidden_size} | num_layers={num_layers} | n_axis={n_axis} | lr={learning_rate:.1e}")
        print(f"{'='*80}")

        fold_results = []

        # Each set of hyperparameters evaluated for each fold
        for val_fold in range(n_folds):
            # Take the validation dataset for this round (all datasets of the held-out atmosphere)
            val_data = data[val_fold] if isinstance(data[val_fold], list) else [data[val_fold]]
            
            # Create the training folds (flatten datasets from all remaining atmospheres)
            train_data = []
            for i in range(n_folds):
                if i != val_fold:
                    if isinstance(data[i], list):
                        train_data.extend(data[i])
                    else:
                        train_data.append(data[i])

            # Print fold being evaluated
            print(f"\n  ↳ [Fold {val_fold + 1}/{n_folds}] Validating on Atmosphere {val_fold + 1} ({len(val_data)} datasets) | Training on {len(train_data)} datasets")

            # Create datasets
            val_dataset   = SlopesDataset(val_data, past_horizon, pred_horizon=pred_horizon)
            train_dataset = SlopesDataset(train_data, past_horizon, pred_horizon=pred_horizon)

            # Create dataloaders with mini-batching for accelerated training
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,  # Shuffle is OK at the level of temporal windows
                pin_memory=pin_memory,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=pin_memory,
            )
            
            # Create model, optimizer and scheduler
            model = IndependentSlopeLSTM(n_axis=n_axis, hidden_size=hidden_size, num_layers=num_layers).to(device)
            optim = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            min_delta = 1e-4
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim, T_max=max_epochs, eta_min=1e-6
            )

            # Train: Error metric is mean square error (MSE)
            criterion = torch.nn.MSELoss()

            early_stopping_patience = 20

            best_val_loss = np.inf
            best_train_loss = np.inf
            best_epoch = 0
            epochs_without_improvement = 0

            for epoch in range(max_epochs):
                # Training
                model.train()
                total_train_loss = 0.0
                n_train_samples = 0

                for x, y_truth in train_loader:
                    # Move data to device
                    x = x.to(device)
                    y_truth = y_truth.to(device)

                    # Reset gradients
                    optim.zero_grad()

                    # Predict / forward
                    y_pred = model(x)

                    # Compute loss
                    loss = criterion(y_pred, y_truth)

                    # Backpropagate the loss
                    loss.backward()

                    # Update model weights
                    optim.step()

                    # Accumulate train loss weighted by batch size
                    batch_len = len(x)
                    total_train_loss += loss.item() * batch_len
                    n_train_samples += batch_len

                # Compute average train loss
                train_loss = total_train_loss / max(n_train_samples, 1)

                # Validation
                model.eval()
                total_val_loss = 0.0
                n_val_samples = 0

                with torch.no_grad():
                    for x, y_truth in val_loader:
                        x = x.to(device)
                        y_truth = y_truth.to(device)

                        y_pred = model(x)
                        loss = criterion(y_pred, y_truth)

                        batch_len = len(x)
                        total_val_loss += loss.item() * batch_len
                        n_val_samples += batch_len

                val_loss = total_val_loss / max(n_val_samples, 1)

                # Update learning rate with cosine annealing
                scheduler.step()

                # Check early stopping
                if val_loss < best_val_loss * (1 - min_delta):
                    best_val_loss = val_loss
                    best_train_loss = train_loss
                    best_epoch = epoch + 1
                    epochs_without_improvement = 0
                    is_best = True
                else:
                    epochs_without_improvement += 1
                    is_best = False

                # Log results every 10 epochs or on epoch 1
                if (epoch + 1) == 1 or (epoch + 1) % 10 == 0:
                    star = " *" if is_best else ""
                    print(
                        f"      Epoch {epoch + 1:3d}/{max_epochs} | "
                        f"Train: {train_loss:.5e} | "
                        f"Val: {val_loss:.5e}{star} | "
                        f"LR: {optim.param_groups[0]['lr']:.2e}"
                    )

                # Early stopping
                if epochs_without_improvement >= early_stopping_patience:
                    print(f"      [Early stopping at epoch {epoch + 1}]")
                    break

            print(f"  ✓ [Fold {val_fold + 1}/{n_folds} Done] Best Epoch: {best_epoch:3d} | Best Val Loss: {best_val_loss:.6e} | Train Loss: {best_train_loss:.6e}")

            fold_results.append({
                "fold": val_fold + 1,
                "train_loss": best_train_loss,
                "val_loss": best_val_loss,
                "best_epoch": best_epoch,
            })

            # Clean up GPU memory per fold
            del model, optim, scheduler
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Compute average parameter for this set of hyperparameters
        val_losses = [result["val_loss"] for result in fold_results]

        mean_val_loss = float(np.mean(val_losses))
        std_val_loss = float(np.std(val_losses))

        print(f"\n  >> Summary Config [{comb_idx}/{n_combs}]: Mean Val Loss = {mean_val_loss:.6e} +/- {std_val_loss:.6e}")

        # Save result for the set of hyperparameters         
        results.append({
            "past_horizon": past_horizon,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "n_axis": n_axis,
            "learning_rate": learning_rate,
            "mean_val_loss": mean_val_loss,
            "std_val_loss": std_val_loss,
            "fold_results": fold_results,
        })            

    return results

# Entry point
if __name__ == "__main__":
    # Reproducibility seed (commented out for now)
    # torch.manual_seed(42)
    # np.random.seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # Training parameters (ordered from highest to lowest GPU memory usage)
    past_horizon = [36, 24, 16, 8, 4]
    hidden_size = [64, 32, 16]
    num_layers = [2, 1]
    n_axis = [1, 2]
    max_epochs = 200
    batch_size = 16
    pred_horizon = 2

    dalia_path = '/net/dalia/scratch1/nlinares/results/results/predictor/training/training_val'
    local_path = '/home/nlinares/simulations/results/predictorSAOS/training/training_val'

    if os.path.exists(dalia_path):
        base_path = dalia_path
    elif os.path.exists(local_path):
        base_path = local_path
    else:
        base_path = dalia_path

    print(f"Data directory: {base_path}\n")

    # Read the files and load data grouped by atmosphere (5 atmospheres x 4 simulation datasets)
    filenames_by_atm = [
        # Atmosphere 1 (draw 1 and draw 2, without and with vibrations)
        [
            f'{base_path}/res_atm1_draw1.h5',
            f'{base_path}/res_atm1_draw1_vib1.h5',
            f'{base_path}/res_atm1_draw2.h5',
            f'{base_path}/res_atm1_draw2_vib1.h5',
        ],
        # Atmosphere 2
        [
            f'{base_path}/res_atm2_draw1.h5',
            f'{base_path}/res_atm2_draw1_vib2.h5',
            f'{base_path}/res_atm2_draw2.h5',
            f'{base_path}/res_atm2_draw2_vib2.h5',
        ],
        # Atmosphere 3
        [
            f'{base_path}/res_atm3_draw1.h5',
            f'{base_path}/res_atm3_draw1_vib3.h5',
            f'{base_path}/res_atm3_draw2.h5',
            f'{base_path}/res_atm3_draw2_vib3.h5',
        ],
        # Atmosphere 4
        [
            f'{base_path}/res_atm4_draw1.h5',
            f'{base_path}/res_atm4_draw1_vib4.h5',
            f'{base_path}/res_atm4_draw2.h5',
            f'{base_path}/res_atm4_draw2_vib4.h5',
        ],
        # Atmosphere 5
        [
            f'{base_path}/res_atm5_draw1.h5',
            f'{base_path}/res_atm5_draw1_vib5.h5',
            f'{base_path}/res_atm5_draw2.h5',
            f'{base_path}/res_atm5_draw2_vib5.h5',
        ],
    ]
    
    print("Loading simulation datasets...")
    data_by_atm = []
    for atm_idx, atm_files in enumerate(filenames_by_atm):
        atm_data = []
        for fpath in atm_files:
            atm_data.append(load_h5(fpath))
        data_by_atm.append(atm_data)
        print(f"  Atmosphere {atm_idx + 1}: loaded {len(atm_data)} datasets")
    print()
    
    # Create param grid
    param_grid = {
        "past_horizon": past_horizon,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "n_axis": n_axis,
        "learning_rate": [3e-4],
    }

    # Generate all the configurations
    combinations = list(
        itertools.product(
            param_grid["past_horizon"],
            param_grid["hidden_size"],
            param_grid["num_layers"],
            param_grid["n_axis"],
            param_grid["learning_rate"],
        )
    )

    # Begin the training
    results = train_per_fold(
        data_by_atm,
        combinations,
        device,
        max_epochs=max_epochs,
        batch_size=batch_size,
        pred_horizon=pred_horizon,
    )

    # Sort results from best to worst
    results_sorted = sorted(results, key=lambda x: x["mean_val_loss"])

    print("\nResults:")
    print("-" * 100)

    for result in results_sorted:
        print(
            f"past_horizon={result['past_horizon']:2d} | "
            f"hidden_size={result['hidden_size']:2d} | "
            f"num_layers={result['num_layers']} | "
            f"n_axis={result['n_axis']} | "
            f"lr={result['learning_rate']:.1e} | "
            f"val_loss={result['mean_val_loss']:.6e} "
            f"+/- {result['std_val_loss']:.6e}"
        )
    # Show best results
    best_result = min(results, key=lambda x: x["mean_val_loss"])

    print("\nBest hyperparameter configuration:")
    print(f"past_horizon : {best_result['past_horizon']}")
    print(f"hidden_size  : {best_result['hidden_size']}")
    print(f"num_layers   : {best_result['num_layers']}")
    print(f"n_axis       : {best_result['n_axis']}")
    print(f"learning_rate: {best_result['learning_rate']}")
    print(f"Validation   : {best_result['mean_val_loss']:.6e} +/- {best_result['std_val_loss']:.6e}")        

    # Show individual results of the winner set of hyperparameters
    print("\nFold results:")

    for fold_result in best_result["fold_results"]:
        print(
            f"Fold {fold_result['fold']} | "
            f"Val loss: {fold_result['val_loss']:.6e} | "
            f"Train loss: {fold_result['train_loss']:.6e} | "
            f"Best epoch: {fold_result['best_epoch']}"
        )