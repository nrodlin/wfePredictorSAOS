import pathlib
import h5py
import torch
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset, DataLoader

from wfePredictorSAOS.predictor.slopesDataset import SlopesDataset
from wfePredictorSAOS.predictor.lstmModel import SlopesLSTM

def main(nSlopes, train_ds, val_ds, lr, patience, weight_decay, n_epochs, device):
    # Build model
    model, optim, scheduler = build_model(nSlopes, lr, weight_decay, device)    

    # Load data
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2,
        pin_memory=True, persistent_workers=True)
    
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2,
        pin_memory=True, persistent_workers=True) 

    # Train
    print(f"\n==== Train samples: {len(train_ds)} | Val samples: {len(val_ds)} ====")

    min_delta = 1e-4

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(1, n_epochs+1):
        loss_train = train_one_epoch(model, train_loader, optim, device)
        loss_eval = eval_one_epoch(model, val_loader, device)


        # LR on plateau (validation loss as metric)
        scheduler.step(loss_eval)
        lr = optim.param_groups[0]["lr"]

        improved = loss_eval < (best_val - min_delta)
        if improved:
            best_val = loss_eval
            best_epoch = epoch
            epochs_no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, f"best_model_LSTM.pt")
        else:
            epochs_no_improve += 1

        print(f"epoch={epoch:02d}/{n_epochs}"
              f"train={loss_train:.6f} val={loss_eval:.6f} best_val={best_val:.6f} "
              f"lr={lr:.2e} no_improve={epochs_no_improve}/{patience}")

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}. Best epoch={best_epoch} best_val={best_val:.6f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)        

       
    return True

def make_datasets(train_data, val_data, past_horizon, future_horizon, stride):
    # First make the training datasets
    train_ds = ConcatDataset([SlopesDataset(ds, past_horizon, future_horizon, stride) for ds in train_data])
    val_ds = ConcatDataset([SlopesDataset(ds, past_horizon, future_horizon, stride) for ds in val_data])

    return train_ds, val_ds

def build_model(nSlopes, lr, weight_decay, device):
    model = SlopesLSTM(nSlopes).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim,
        mode="min",
        factor=0.5,
        patience=2,         # epochs without improving before reducing LR
        threshold=1e-4,     # minimum variation to consider that the model learnt
        threshold_mode="rel",
        min_lr=1e-7,
    )

    return model, optim, scheduler   

def train_one_epoch(model, loader, optim, device):
    model.train()

    total_loss = 0
    n = 0

    for x, y in loader:
        x = x.to(device) # batch_size, past_horizon, nSlopes
        y = y.to(device) # batch_size, nSlopes (ground-truth for the prediction)
        # Make the prediction
        pred = model(x)
        # Compute the error
        loss = torch.nn.functional.mse_loss(pred, y)
        # Reset the gradients to avoid cummulating old data
        optim.zero_grad(set_to_none=True)
        # Backpropagate the error to compute the gradients
        loss.backward()
        # Update the model using the gradients
        optim.step()

        # Store error
        bs = x.size(0)
        total_loss += loss.item()*bs
        n +=bs
    
    return total_loss / max(n, 1)

@torch.no_grad()
def eval_one_epoch(model, loader, device):
    model.eval()

    total_loss = 0
    n = 0

    for x, y in loader:
        x = x.to(device) # batch_size, past_horizon, nSlopes
        y = y.to(device) # batch_size, nSlopes (ground-truth for the prediction)
        # Make the prediction
        pred = model(x)
        # Compute the error
        loss = torch.nn.functional.mse_loss(pred, y)
        # Store error
        bs = x.size(0)
        total_loss += loss.item()*bs
        n +=bs
    
    return total_loss / max(n, 1)
        
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    path_to_train_datasets = 'C:/Users/nicolas/Downloads/trainingData'
    path_to_val_datasets = 'C:/Users/nicolas/Downloads/validationData'
    past_horizon = 8
    future_horizon = 2
    stride = 2

    n_epochs = 250
    lr = 1e-4
    weight_decay = 1e-3
    patience = 5
    

    # Load data from h5 files
    trainingData = pathlib.Path(path_to_train_datasets)
    validationData = pathlib.Path(path_to_val_datasets)

    list_datasets_train = [str(p) for p in trainingData.rglob('*.h5')]
    list_datasets_val   = [str(p) for p in validationData.rglob('*.h5')]

    dataset_list_train = []
    for fp in list_datasets_train:
        with h5py.File(fp, 'r') as f:
            dataset_list_train.append(torch.from_numpy(
                f['LightPath_0']['slopes_1D']['data'][:].squeeze()
            ).float())

    dataset_list_val = []
    for fp in list_datasets_val:
        with h5py.File(fp, 'r') as f:
            dataset_list_val.append(torch.from_numpy(
                f['LightPath_0']['slopes_1D']['data'][:].squeeze()
            ).float())
    
    nSlopes  = dataset_list_val[0].shape[-1]

    # Make the datasets
    train_ds, val_ds = make_datasets(dataset_list_train, dataset_list_val, past_horizon, 
                                     future_horizon, stride)
    # Train the model
    main(nSlopes, train_ds, val_ds, lr, patience, weight_decay, n_epochs, device)
