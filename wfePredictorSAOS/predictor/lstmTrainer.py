import pathlib
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from wfePredictorSAOS.predictor.slopesDataset import SlopesDataset
from wfePredictorSAOS.predictor.independentLSTMModel import IndependentSlopeLSTM

# ==============================================================================
# NORMALIZACIÓN
# ==============================================================================
def apply_normalization_per_file(dataset_list, label=""):
    print(f"Normalizando {label} por fichero:")
    normalized, stats = [], []
    for i, d in enumerate(dataset_list):
        m = d.mean(dim=0)
        s = d.std(dim=0).clamp(min=1e-6)
        normalized.append((d - m) / s)
        stats.append((m, s))
        print(f"  fichero {i:02d}: mean={m.mean():.4f}  std={s.mean():.4f}  "
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
# MODELO, OPTIMIZADOR Y SCHEDULER
# ==============================================================================
def build_model(hidden_size, lr, weight_decay, device):
    model = IndependentSlopeLSTM(hidden_size=hidden_size, num_layers=1).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] Parámetros entrenables: {n_params:,}\n")
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
# BUCLE PRINCIPAL
# ==============================================================================
def train(hidden_size, train_ds, val_ds, lr, patience, weight_decay,
          n_epochs, device, baseline_mse):
    model, optim, scheduler = build_model(hidden_size, lr, weight_decay, device)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,
                              num_workers=2, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False,
                              num_workers=2, pin_memory=True, persistent_workers=True)

    print(f"==== Train samples: {len(train_ds)} | Val samples: {len(val_ds)} ====")
    print(f"     Baseline MSE a superar: {baseline_mse:.6f}\n")

    best_val, best_state, best_epoch = float("inf"), None, 0
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
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            torch.save(best_state, "best_model_IndepLSTM.pt")
        else:
            epochs_no_improve += 1

        beating = "✅" if loss_eval < baseline_mse * 0.95 else "⚠️ "
        print(f"epoch={epoch:03d}/{n_epochs} "
              f"train={loss_train:.6f} val={loss_eval:.6f} "
              f"best={best_val:.6f} {beating} "
              f"lr={current_lr:.2e} no_improve={epochs_no_improve}/{patience}")

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping en epoch {epoch}. "
                  f"Mejor epoch={best_epoch}, mejor val={best_val:.6f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, train_losses, val_losses


# ==============================================================================
# VISUALIZACIÓN
# ==============================================================================
@torch.no_grad()
def collect_predictions(model, val_ds, device, n_samples=5):
    """
    Recoge n_samples aleatorios del val set y devuelve:
      - x_seq   : (n_samples, past_horizon, nSlopes)  — historia de entrada
      - y_true  : (n_samples, nSlopes)                — verdad
      - y_pred  : (n_samples, nSlopes)                — predicción del modelo
    """
    model.eval()
    indices = torch.randperm(len(val_ds))[:n_samples]
    xs, ys_true, ys_pred = [], [], []
    for i in indices:
        x, y = val_ds[i]
        xs.append(x)
        ys_true.append(y)
        pred = model(x.unsqueeze(0).to(device)).squeeze(0).cpu()
        ys_pred.append(pred)
    return (torch.stack(xs),
            torch.stack(ys_true),
            torch.stack(ys_pred))


def plot_training_curve(train_losses, val_losses, best_epoch):
    fig, ax = plt.subplots(figsize=(9, 4))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train loss", linewidth=1.5)
    ax.plot(epochs, val_losses,   label="Val loss",   linewidth=1.5)
    ax.axvline(best_epoch, color="green", linestyle="--",
               linewidth=1, label=f"Best epoch ({best_epoch})")
    ax.set_xlabel("Época")
    ax.set_ylabel("MSE")
    ax.set_title("Curva de entrenamiento")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("training_curve.png", dpi=150)
    plt.show()
    print("[PLOT] training_curve.png guardado")


def plot_slope_predictions(x_seq, y_true, y_pred, n_slopes_shown=8,
                           sample_idx=0, future_horizon=2):
    """
    Para una muestra concreta, muestra n_slopes_shown slopes individuales:
    - Historia pasada (línea azul)
    - Verdad en t+future_horizon (punto verde)
    - Predicción (punto rojo)
    """
    past  = x_seq[sample_idx].numpy()    # (past_horizon, nSlopes)
    truth = y_true[sample_idx].numpy()   # (nSlopes,)
    pred  = y_pred[sample_idx].numpy()   # (nSlopes,)

    T = past.shape[0]
    t_past = np.arange(T)
    t_pred = T + future_horizon - 1      # índice temporal de la predicción

    # Elige slopes distribuidas uniformemente a lo largo del vector
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
                linewidth=1.5, marker="o", markersize=3, label="Historia")
        ax.scatter(t_pred, truth[s], color="green",
                   s=80, zorder=5, label="Verdad")
        ax.scatter(t_pred, pred[s],  color="red",
                   s=80, marker="x", zorder=5, linewidths=2, label="Predicción")
        # Error
        error = abs(truth[s] - pred[s])
        ax.set_title(f"Slope {s}  |  |err|={error:.4f}", fontsize=9)
        ax.axvline(T - 1, color="gray", linestyle=":", linewidth=0.8)
        ax.set_xlabel("Frame (normalizado)")
        if plot_i == 0:
            ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Oculta ejes sobrantes
    for j in range(plot_i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Predicción por slope — muestra val #{sample_idx}", fontsize=12)
    plt.tight_layout()
    fname = f"predictions_per_slope_sample{sample_idx}.png"
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"[PLOT] {fname} guardado")


def plot_full_vector_comparison(y_true, y_pred, nSlopes, n_samples=3):
    """
    Compara el vector completo de slopes (verdad vs predicción) para
    n_samples muestras. Útil para ver si hay patrones sistemáticos en el error.
    """
    fig, axes = plt.subplots(n_samples, 1,
                             figsize=(14, 3.5 * n_samples),
                             sharex=True)
    if n_samples == 1:
        axes = [axes]

    slope_axis = np.arange(nSlopes)
    half = nSlopes // 2  # separación entre slopes X e Y

    for i, ax in enumerate(axes):
        truth = y_true[i].numpy()
        pred  = y_pred[i].numpy()
        error = truth - pred

        ax.plot(slope_axis, truth, color="green",    linewidth=0.8,
                alpha=0.9, label="Verdad")
        ax.plot(slope_axis, pred,  color="red",      linewidth=0.8,
                alpha=0.9, label="Predicción")
        ax.fill_between(slope_axis, 0, error,        color="orange",
                        alpha=0.4, label="Error")
        ax.axvline(half, color="gray", linestyle="--", linewidth=1,
                   label="X | Y")
        ax.set_ylabel("Amplitud (norm.)")
        ax.set_title(f"Muestra val #{i}  —  MSE={np.mean(error**2):.6f}",
                     fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Índice slope  (izq: X,  der: Y)")
    fig.suptitle("Vector completo de slopes: verdad vs predicción", fontsize=13)
    plt.tight_layout()
    plt.savefig("full_vector_comparison.png", dpi=150)
    plt.show()
    print("[PLOT] full_vector_comparison.png guardado")


def plot_error_distribution(y_true, y_pred):
    """
    Histograma del error por slope sobre todas las muestras de val.
    Muestra si el error es sistemático o aleatorio.
    """
    error = (y_true - y_pred).numpy().flatten()
    mse   = np.mean(error ** 2)
    mae   = np.mean(np.abs(error))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histograma del error
    axes[0].hist(error, bins=80, color="steelblue", edgecolor="none", alpha=0.8)
    axes[0].axvline(0, color="red", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Error (pred - verdad)")
    axes[0].set_ylabel("Frecuencia")
    axes[0].set_title(f"Distribución del error\nMSE={mse:.6f}  MAE={mae:.6f}")
    axes[0].grid(True, alpha=0.3)

    # Error medio por slope (¿hay slopes sistemáticamente malas?)
    error_2d     = (y_true - y_pred).numpy()         # (n_samples, nSlopes)
    mean_abs_err = np.abs(error_2d).mean(axis=0)     # (nSlopes,)
    axes[1].plot(mean_abs_err, linewidth=0.8, color="darkorange")
    axes[1].axvline(mean_abs_err.shape[0] // 2, color="gray",
                    linestyle="--", linewidth=1, label="X | Y")
    axes[1].set_xlabel("Índice slope")
    axes[1].set_ylabel("|Error| medio")
    axes[1].set_title("Error absoluto medio por slope")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("error_distribution.png", dpi=150)
    plt.show()
    print("[PLOT] error_distribution.png guardado")


def plot_temporal_sequence(model, val_data_raw, device, past_horizon,
                           future_horizon, n_frames=80, slope_indices=None):
    """
    Toma un fichero de val completo y muestra la evolución temporal de
    algunas slopes: historia continua + predicción frame a frame.
    Permite ver si el modelo sigue la dinámica temporal.
    """
    if slope_indices is None:
        nS = val_data_raw.shape[1]
        slope_indices = np.linspace(0, nS - 1, 6, dtype=int)

    model.eval()
    data = val_data_raw  # (N, nSlopes), ya normalizado

    preds, truths = [], []
    with torch.no_grad():
        for t in range(n_frames):
            start = t
            end   = t + past_horizon
            if end + future_horizon > len(data):
                break
            x    = data[start:end].unsqueeze(0).to(device)   # (1, T, nSlopes)
            pred = model(x).squeeze(0).cpu()
            truth = data[end + future_horizon - 1]
            preds.append(pred)
            truths.append(truth)

    preds  = torch.stack(preds).numpy()   # (n_frames, nSlopes)
    truths = torch.stack(truths).numpy()  # (n_frames, nSlopes)
    t_axis = np.arange(len(preds))

    ncols = 3
    nrows = int(np.ceil(len(slope_indices) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 5, nrows * 3),
                             sharey=False)
    axes = axes.flatten()

    for plot_i, s in enumerate(slope_indices):
        ax = axes[plot_i]
        ax.plot(t_axis, truths[:, s], color="green",
                linewidth=1.2, label="Verdad", alpha=0.9)
        ax.plot(t_axis, preds[:, s],  color="red",
                linewidth=1.0, label="Predicción", alpha=0.9, linestyle="--")
        corr = np.corrcoef(truths[:, s], preds[:, s])[0, 1]
        ax.set_title(f"Slope {s}  |  r={corr:.3f}", fontsize=9)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Amplitud (norm.)")
        if plot_i == 0:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for j in range(plot_i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Evolución temporal: verdad vs predicción (fichero val 0)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig("temporal_sequence.png", dpi=150)
    plt.show()
    print("[PLOT] temporal_sequence.png guardado")


# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}\n")

    # ── Rutas ─────────────────────────────────────────────────────────────────
    path_to_train_datasets = 'C:/Users/nicolas/Downloads/trainingData'
    path_to_val_datasets   = 'C:/Users/nicolas/Downloads/validationData'

    # ── Hiperparámetros ───────────────────────────────────────────────────────
    past_horizon   = 8
    future_horizon = 2
    stride         = 1
    n_epochs       = 250
    lr             = 1e-4
    weight_decay   = 1e-2
    patience       = 10
    hidden_size    = 32

    # ── Carga de datos ────────────────────────────────────────────────────────
    list_datasets_train = [str(p) for p in pathlib.Path(path_to_train_datasets).rglob('*.h5')]
    list_datasets_val   = [str(p) for p in pathlib.Path(path_to_val_datasets).rglob('*.h5')]

    dataset_list_train, dataset_list_val = [], []
    for fp in list_datasets_train:
        with h5py.File(fp, 'r') as f:
            dataset_list_train.append(
                torch.from_numpy(
                    f['LightPath_0']['slopes_1D']['data'][:].squeeze()
                ).float()
            )
    for fp in list_datasets_val:
        with h5py.File(fp, 'r') as f:
            dataset_list_val.append(
                torch.from_numpy(
                    f['LightPath_0']['slopes_1D']['data'][:].squeeze()
                ).float()
            )

    nSlopes = dataset_list_train[0].shape[-1]
    print(f"nSlopes = {nSlopes}\n")

    # ── Normalización ─────────────────────────────────────────────────────────
    dataset_list_train, _          = apply_normalization_per_file(dataset_list_train, "train")
    dataset_list_val,   val_stats  = apply_normalization_per_file(dataset_list_val,   "val")

    # ── Baseline ──────────────────────────────────────────────────────────────
    all_train    = torch.cat(dataset_list_train, dim=0)
    baseline_mse = all_train.var(dim=0).mean().item()
    print(f"[CHECK] Baseline MSE: {baseline_mse:.6f}\n")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds, val_ds = make_datasets(
        dataset_list_train, dataset_list_val,
        past_horizon, future_horizon, stride,
    )

    # ── Entrenamiento ─────────────────────────────────────────────────────────
    model, train_losses, val_losses = train(
        hidden_size, train_ds, val_ds,
        lr, patience, weight_decay, n_epochs,
        device, baseline_mse,
    )

    # ── Determina la mejor época para el plot ─────────────────────────────────
    best_epoch = int(np.argmin(val_losses)) + 1

    # ==========================================================================
    # VISUALIZACIONES
    # ==========================================================================

    # 1. Curva de entrenamiento
    plot_training_curve(train_losses, val_losses, best_epoch)

    # 2. Recoge predicciones sobre el val set
    x_seq, y_true, y_pred = collect_predictions(model, val_ds, device, n_samples=5)

    # 3. Predicción slope a slope para una muestra concreta
    #    Cambia sample_idx para ver otras muestras (0..4)
    plot_slope_predictions(x_seq, y_true, y_pred,
                           n_slopes_shown=8,
                           sample_idx=0,
                           future_horizon=future_horizon)

    # 4. Vector completo verdad vs predicción
    plot_full_vector_comparison(y_true, y_pred, nSlopes, n_samples=3)

    # 5. Distribución del error (¿sistemático o aleatorio?)
    #    Usa todas las muestras disponibles para más estadística
    x_all, y_true_all, y_pred_all = collect_predictions(
        model, val_ds, device, n_samples=min(200, len(val_ds))
    )
    plot_error_distribution(y_true_all, y_pred_all)

    # 6. Evolución temporal continua sobre el primer fichero de val
    #    Muestra si el modelo sigue la dinámica frame a frame
    plot_temporal_sequence(
        model,
        val_data_raw=dataset_list_val[0],   # primer fichero val, ya normalizado
        device=device,
        past_horizon=past_horizon,
        future_horizon=future_horizon,
        n_frames=150,
        slope_indices=None,                  # None = distribuidas uniformemente
    )