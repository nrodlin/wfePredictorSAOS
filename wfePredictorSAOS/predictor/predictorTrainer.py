import pathlib
import h5py
import torch
import matplotlib.pyplot as plt
from wfePredictorSAOS.predictor.tcnPredictor import SHWFS_TCN, WindowDataset, train_one_epoch, eval_one_epoch
from torch.utils.data import ConcatDataset

def build_model(nSubap, device):
    model = SHWFS_TCN(n_subap=nSubap, emb=128, tcn_layers=4, tcn_kernel=3, dropout=0.1).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim,
        mode="min",
        factor=0.5,
        patience=2,         # epochs sin mejora para bajar LR
        threshold=1e-4,     # similar a tu min_delta
        threshold_mode="rel",
        min_lr=1e-7,
    )
    return model, optim, scheduler

def main(dataset_list_train, dataset_list_val, nSubap, past_windows, future_horizon, mask2, stride, n_epochs, device):

    for T in past_windows:

        dataset_objects_train = [
            WindowDataset(ds, nSubap, T, future_horizon, stride)
            for ds in dataset_list_train
        ]
        train_ds = ConcatDataset(dataset_objects_train)

        dataset_objects_val = [
            WindowDataset(ds, nSubap, T, future_horizon, 1)
            for ds in dataset_list_val
        ]
        val_ds = ConcatDataset(dataset_objects_val)

        model, optim, scheduler = build_model(nSubap, device)

        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=64, shuffle=True, num_workers=2,
            pin_memory=True, persistent_workers=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=64, shuffle=False, num_workers=2,
            pin_memory=True, persistent_workers=True
        )

        print(f"\n==== T={T} | train_samples={len(train_ds)} | val_samples={len(val_ds)} ====")

        patience = 5
        min_delta = 1e-4

        best_val = float("inf")
        best_state = None
        best_epoch = 0
        epochs_no_improve = 0

        for epoch in range(1, n_epochs + 1):
            tr = train_one_epoch(model, train_loader, optim, device=device, mask2=mask2)
            va = eval_one_epoch(model, val_loader, device=device, mask2=mask2)

            # LR on plateau (usa la val como métrica)
            scheduler.step(va)
            lr = optim.param_groups[0]["lr"]

            improved = va < (best_val - min_delta)
            if improved:
                best_val = va
                best_epoch = epoch
                epochs_no_improve = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save(best_state, f"best_model_T{T}.pt")
            else:
                epochs_no_improve += 1

            print(
                f"T={T} epoch={epoch:02d}/{n_epochs} "
                f"train={tr:.6f} val={va:.6f} best_val={best_val:.6f} "
                f"lr={lr:.2e} no_improve={epochs_no_improve}/{patience}"
            )

            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}. Best epoch={best_epoch} best_val={best_val:.6f}")
                break

        if best_state is not None:
            model.load_state_dict(best_state)

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    path_to_train_datasets = 'C:/Users/nicolas/Downloads/trainingData'
    path_to_val_datasets = 'C:/Users/nicolas/Downloads/validationData'
    past_windows = [8]
    future_horizon = 2
    n_epochs = 20
    stride = 2

    trainingData = pathlib.Path(path_to_train_datasets)
    validationData = pathlib.Path(path_to_val_datasets)

    list_datasets_train = [str(p) for p in trainingData.rglob('*.h5')]
    list_datasets_val   = [str(p) for p in validationData.rglob('*.h5')]

    dataset_list_train = []
    for fp in list_datasets_train:
        with h5py.File(fp, 'r') as f:
            dataset_list_train.append(torch.from_numpy(
                f['LightPath_0']['slopes_2D']['data'][:].squeeze()
            ).float())

    dataset_list_val = []
    for fp in list_datasets_val:
        with h5py.File(fp, 'r') as f:
            dataset_list_val.append(torch.from_numpy(
                f['LightPath_0']['slopes_2D']['data'][:].squeeze()
            ).float())

    nSubap = dataset_list_train[0].shape[-1]
    H = nSubap

    # mask (2H,W) -> mask2 (2,H,W)
    pupil_mask2H = (dataset_list_train[0].sum(dim=0) != 0)  # bool (2H,W)
    mask2 = torch.stack([pupil_mask2H[:H, :], pupil_mask2H[H:, :]], dim=0).float()

    main(dataset_list_train, dataset_list_val, nSubap, past_windows, future_horizon, mask2, stride, n_epochs, device)