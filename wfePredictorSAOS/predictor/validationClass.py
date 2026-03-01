import torch
import torch.nn.functional as F

def mask_2H_to_2HW(mask2H_W, nSubap, device):
    if not isinstance(mask2H_W, torch.Tensor):
        mask2H_W = torch.from_numpy(mask2H_W)
    mask2H_W = mask2H_W.float().to(device)
    H = nSubap
    mX = mask2H_W[:H, :]
    mY = mask2H_W[H:, :]
    return torch.stack([mX, mY], dim=0)  # (2,H,W)

@torch.no_grad()
def eval_sequence_T8(model, seq_real, nSubap, horizon=2, mask2H_W=None, device="cuda"):
    """
    seq_real: (N, 2H, W) torch tensor
    Returns:
      mse_model_mean, mse_base_mean, errors_model, errors_base
    """
    model.eval()
    H = nSubap
    T = 8
    N = seq_real.shape[0]

    seq = seq_real.to(device)

    mask2 = None
    if mask2H_W is not None:
        mask2 = mask_2H_to_2HW(mask2H_W, nSubap, device)  # (2,H,W)

    errors_model = []
    errors_base = []

    for t in range(T - 1, N - horizon):
        window = seq[t - T + 1 : t + 1]  # (T, 2H, W)

        # Build model input (1,T,2,H,W)
        xX = window[:, :H, :]
        xY = window[:, H:, :]
        x = torch.stack([xX, xY], dim=1)  # (T,2,H,W)

        # Target (2,H,W)
        target_2H_W = seq[t + horizon]
        yX = target_2H_W[:H, :]
        yY = target_2H_W[H:, :]
        y = torch.stack([yX, yY], dim=0)  # (2,H,W)

        # Baseline: persistence y_hat = x_t
        base_2H_W = window[-1]  # (2H,W) = slopes at time t
        bX = base_2H_W[:H, :]
        bY = base_2H_W[H:, :]
        b = torch.stack([bX, bY], dim=0)  # (2,H,W)

        # Apply mask (recommended)
        if mask2 is not None:
            x = x * mask2.unsqueeze(0)
            y = y * mask2
            b = b * mask2

        # Predict
        pred = model(x.unsqueeze(0)).squeeze(0)  # (2,H,W)

        # Errors (mean over valid elements if masked, else full)
        mse_m = F.mse_loss(pred, y, reduction="mean").item()
        mse_b = F.mse_loss(b, y, reduction="mean").item()

        errors_model.append(mse_m)
        errors_base.append(mse_b)

    mse_model_mean = sum(errors_model) / len(errors_model)
    mse_base_mean  = sum(errors_base) / len(errors_base)
    return mse_model_mean, mse_base_mean, errors_model, errors_base