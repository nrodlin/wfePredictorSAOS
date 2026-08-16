import h5py
import numpy as np
import torch
import itertools

from torch.utils.data import DataLoader

from wfePredictorSAOS.predictor.slopesDataset import SlidingWindowDataset
from wfePredictorSAOS.predictor.independentLSTMModel import IndependentSlopeLSTM

### Load h5 File

def load_h5(filename):
    with h5py.File(filename, 'r') as f:
            data = torch.from_numpy(f['LightPath_0']['slopes_1D']['data'][:].squeeze()).float()
    return data

# launch the training
def train_per_fold(data, combinations, device, max_epochs):

    n_folds = len(data)

    results = []

    # Iterate with the iterator
    for combination in combinations:
        # Extract each hyperparameter from the combination
        past_horizon, hidden_size, num_layers, n_axis, learning_rate = combination

        fold_results = []

        # Each set of hyperparameters evaluated for each fold
        for val_fold in range(n_folds):
            # Take the validation dataset for this round
            val_data = [data[val_fold]]
            # Create the trainning folds
            train_data = [data[i] for i in range(n_folds) if i != val_fold]

            # Print combination being evaluated
            print(f'Combination {combination} | Validation fold {val_fold}')

            # Create datasets
            val_dataset   = SlidingWindowDataset(val_data, past_horizon, pred_horizon=2)
            train_dataset = SlidingWindowDataset(train_data, past_horizon, pred_horizon=2)

            # Create dataloaders
            train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True) # Shuffle is OK, because it is at the level of coherent temporal windows, high level batch
                                                                                    # is None because internally, each slope vector is treated as a batch of predictions
            val_loader = DataLoader(val_dataset, batch_size=None, shuffle=False) # For the validation, shuffle is not relevant
            
            # Create model, optimizer and scheduler
            model = IndependentSlopeLSTM(n_axis=n_axis, hidden_size=hidden_size, num_layers=num_layers).to(device)
            optim = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
            min_delta = 1e-4
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=0.5, patience=8, threshold=min_delta, threshold_mode="rel", min_lr=1e-7)

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
                train_loss = 0.0

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

                    # Add train loss
                    train_loss += loss.item()

                # Compute average train loss
                train_loss /= len(train_loader)

                # Validation
                model.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for x, y_truth in val_loader:
                        x = x.to(device)
                        y_truth = y_truth.to(device)

                        y_pred = model(x)
                        loss = criterion(y_pred, y_truth)

                        val_loss += loss.item()

                val_loss /= len(val_loader)

                # Update learning rate if on plateau
                scheduler.step(val_loss)

                # Check early stopping
                if val_loss < best_val_loss * (1 - min_delta):
                    best_val_loss = val_loss
                    best_train_loss = train_loss
                    best_epoch = epoch + 1
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                # Log results
                if (epoch % 15) == 0:
                    print(
                        f"Epoch {epoch + 1}/{max_epochs} | "
                        f"Train loss: {train_loss:.6e} | "
                        f"Val loss: {val_loss:.6e} | "
                        f"LR: {optim.param_groups[0]['lr']:.2e}"
                    )

                # Early stopping
                if epochs_without_improvement >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break      
            fold_results.append({"fold": val_fold,
                                "train_loss": best_train_loss,
                                "val_loss": best_val_loss,
                                "best_epoch": best_epoch})

        # Compute average parameter for this set of hyperparameters
        val_losses = [result["val_loss"] for result in fold_results]

        mean_val_loss = np.mean(val_losses)
        std_val_loss = np.std(val_losses)

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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # Training parameters
    past_horizon = [4, 8, 16]
    hidden_size = [16, 32, 64]
    num_layers = [1, 2]
    n_axis = [1, 2]
    max_epochs = 200

    # Read the files and load the data
    filenames = ['atm1.h5',
                 'atm2.h5',
                 'atm3.h5',
                 'atm4.h5',
                 'atm5.h5']
    
    data_list = []

    for i in range(len(filenames)):
        data_list.append(load_h5(filenames[i]))
    
    # Create param grid

    param_grid = {
        "past_horizon": past_horizon,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "n_axis_mode": n_axis,
        "learning_rate": [1e-3],
    }

    # Generate all the configurations
    combinations = list(
        itertools.product(
            param_grid["past_horizon"],
            param_grid["hidden_size"],
            param_grid["num_layers"],
            param_grid["n_axis_mode"],
            param_grid["learning_rate"],
        )
    )

    # Begin the training

    results = train_per_fold(data_list, combinations, device, max_epochs)

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