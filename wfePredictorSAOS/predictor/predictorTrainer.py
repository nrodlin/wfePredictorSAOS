import pathlib
import h5py
import torch
import matplotlib.pyplot as plt
from wfePredictorSAOS.predictor.tcnPredictor import SHWFS_TCN, WindowDataset, train_one_epoch, eval_one_epoch, evaluate_sequence
from torch.utils.data import ConcatDataset

def build_model(nSubap, device):
    model = SHWFS_TCN(n_subap=nSubap, emb=128, tcn_layers=4, tcn_kernel=3, dropout=0.3).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    return model, optim

def main(dataset_list_train, dataset_list_val, nSubap, past_windows, future_horizon, mask_2D, stride, n_epochs, device):
    # (opcional pero recomendable) fija semilla para comparar
    # torch.manual_seed(0)
    # if torch.cuda.is_available():
        # torch.cuda.manual_seed_all(0) 

    for T in past_windows:

        # --- create datasets ---
        dataset_objects_train = [
            WindowDataset(ds, nSubap, T, future_horizon, mask_2D, stride)
            for ds in dataset_list_train
        ]
        train_ds = ConcatDataset(dataset_objects_train)

        dataset_objects_val = [
            WindowDataset(ds, nSubap, T, future_horizon, mask_2D, 1)
            for ds in dataset_list_val
        ]
        val_ds = ConcatDataset(dataset_objects_val)

        # --- reboot model for a fair comparison of T ---
        model, optim = build_model(nSubap, device)

        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=64, shuffle=True, num_workers=2,
            pin_memory=True, persistent_workers=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=64, shuffle=False, num_workers=2,
            pin_memory=True, persistent_workers=True
        )

        print(f"\n==== T={T} | train_samples={len(train_ds)} | val_samples={len(val_ds)} ====")

        # --- early stopping settings (per T) ---
        patience = 2
        min_delta = 1e-4

        best_val = float("inf")
        best_state = None
        best_epoch = 0
        epochs_no_improve = 0

        for epoch in range(1, n_epochs + 1):
            tr = train_one_epoch(model, train_loader, optim, device=device)
            va = eval_one_epoch(model, val_loader, device=device)

            improved = va < (best_val - min_delta)
            if improved:
                best_val = va
                best_epoch = epoch
                epochs_no_improve = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                # optional: save checkpoint per T
                torch.save(best_state, f"best_model_T{T}.pt")
            else:
                epochs_no_improve += 1

            print(
                f"T={T} epoch={epoch:02d}/{n_epochs} "
                f"train={tr:.6f} val={va:.6f} best_val={best_val:.6f} "
                f"no_improve={epochs_no_improve}/{patience}"
            )

            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch}. Best epoch={best_epoch} best_val={best_val:.6f}")
                break

        # Restore best model
        if best_state is not None:
            model.load_state_dict(best_state)

if __name__ == "__main__":

    device = "cpu"#"cuda" if torch.cuda.is_available() else "cpu"
    # Parameters
    path_to_train_datasets = 'C:/Users/nicolas/Downloads/trainingData'
    path_to_val_datasets = 'C:/Users/nicolas/Downloads/validationData'
    past_windows = [8]
    future_horizon = 2
    n_epochs = 10
    stride = 2

    # Load datasets
    trainingData = pathlib.Path(path_to_train_datasets)
    validationData = pathlib.Path(path_to_val_datasets)

    list_datasets_train = list(trainingData.rglob('*.h5'))
    list_datasets_train = [str(list_datasets_train[i]) for i in range(len(list_datasets_train))]

    list_datasets_val = list(validationData.rglob('*.h5'))
    list_datasets_val = [str(list_datasets_val[i]) for i in range(len(list_datasets_val))]    

    dataset_list_train = []

    for i in range(len(list_datasets_train)):
        with h5py.File(list_datasets_train[i], 'r') as f:
            dataset_list_train.append(torch.from_numpy(f['LightPath_0']['slopes_2D']['data'][:].squeeze()).float())

    dataset_list_val = []

    for i in range(len(list_datasets_val)):
        with h5py.File(list_datasets_val[i], 'r') as f:
            dataset_list_val.append(torch.from_numpy(f['LightPath_0']['slopes_2D']['data'][:].squeeze()).float())

    # Get the size of the pupil

    nSubap = dataset_list_train[0].shape[-1]
    pupil_mask  = dataset_list_train[0].sum(axis=0) != 0    

    main(dataset_list_train, dataset_list_val, nSubap, past_windows, future_horizon, pupil_mask, stride, n_epochs, device)