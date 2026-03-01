import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Utils: Causal Conv1d helper
# -----------------------------
class Chomp1d(nn.Module):
    """Trim padding on the right to preserve causality."""
    def __init__(self, chomp):
        super().__init__()
        self.chomp = chomp

    def forward(self, x):
        # x: (B, C, T)
        return x[:, :, :-self.chomp] if self.chomp > 0 else x


class TemporalBlock(nn.Module):
    """
    Typical TCN block: dilated causal Conv1d + residual connection.
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.0):
        super().__init__()
        padding = (kernel_size - 1) * dilation  # left padding (implemented via full padding + chomp)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size,
                               padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size,
                               padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)

        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        # x: (B, C, T) causal
        y = self.conv1(x)
        y = self.chomp1(y)
        y = F.gelu(y)
        y = self.dropout(y)

        y = self.conv2(y)
        y = self.chomp2(y)
        y = F.gelu(y)
        y = self.dropout(y)

        res = x if self.downsample is None else self.downsample(x)
        return y + res


class TemporalConvNet(nn.Module):
    def __init__(self, channels, kernel_size=3, dropout=0.0):
        """
        channels: list like [F, F, F] (number of channels per layer).
        """
        super().__init__()
        layers = []
        for i in range(len(channels) - 1):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            dilation = 2 ** i
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size=kernel_size,
                                        dilation=dilation, dropout=dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# -----------------------------
# Spatial encoder per frame
# -----------------------------
class SpatialEncoder(nn.Module):
    def __init__(self, in_ch=2, base=32, emb=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base, base, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base, 2*base, 3, stride=2, padding=1),  # downsample
            nn.GELU(),
            nn.Conv2d(2*base, 2*base, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # (B, 2*base, 1, 1)
        )
        self.proj = nn.Linear(2*base, emb)

    def forward(self, x):
        # x: (B, 2, H, W)
        h = self.net(x).squeeze(-1).squeeze(-1)  # (B, 2*base)
        return self.proj(h)  # (B, emb)


# -----------------------------
# Full model: CNN + TCN + head
# -----------------------------
class SHWFS_TCN(nn.Module):
    def __init__(self, n_subap, emb=256, tcn_layers=4, tcn_kernel=3, dropout=0.0):
        super().__init__()
        self.n_subap = n_subap
        self.spatial = SpatialEncoder(in_ch=2, base=16, emb=emb)

        # TCN: constant channels emb -> emb
        channels = [emb] + [emb] * tcn_layers
        self.tcn = TemporalConvNet(channels, kernel_size=tcn_kernel, dropout=dropout)

        # Head to map back to (2, H, W)
        self.head = nn.Sequential(
            nn.Linear(emb, emb),
            nn.GELU(),
            nn.Linear(emb, 2 * n_subap * n_subap),
        )

    def forward(self, x):
        """
        x: (B, T, 2*nSubap, nSubap) or (B, T, 2, nSubap, nSubap)
        """
        if x.dim() == 4:
            # (B, T, 2H, W) -> (B, T, 2, H, W)
            B, T, HH, W = x.shape
            H = HH // 2
            xX = x[:, :, :H, :]
            xY = x[:, :, H:, :]
            x = torch.stack([xX, xY], dim=2)  # (B, T, 2, H, W)

        B, T, C, H, W = x.shape  # C=2
        assert H == self.n_subap and W == self.n_subap and C == 2

        # Spatial encoder per frame
        feats = []
        for t in range(T):
            feats.append(self.spatial(x[:, t]))  # (B, emb)
        feats = torch.stack(feats, dim=-1)      # (B, emb, T)

        # Temporal TCN
        z = self.tcn(feats)                     # (B, emb, T)

        # Use last time step (t) to predict t+horizon
        z_last = z[:, :, -1]                    # (B, emb)

        y = self.head(z_last).view(B, 2, H, W)  # (B, 2, H, W)
        return y


# -----------------------------
# Example windowed dataset
# -----------------------------
class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, seq, n_subap, T=8, horizon=2, mask2H_W=None, stride=1):
        """
        seq: torch tensor (N, 2*nSubap, nSubap) with N consecutive frames
        mask2H_W: (2*nSubap, nSubap) float/bool mask (1=valid, 0=invalid)
        """
        super().__init__()
        self.seq = seq
        self.T = T
        self.h = horizon
        self.n_subap = n_subap
        self.stride = stride

        self.mask2 = None
        if mask2H_W is not None:
            if not isinstance(mask2H_W, torch.Tensor):
                mask2H_W = torch.from_numpy(mask2H_W)
            mask2H_W = mask2H_W.float()
            H = n_subap
            mX = mask2H_W[:H, :]
            mY = mask2H_W[H:, :]
            self.mask2 = torch.stack([mX, mY], dim=0)  # (2,H,W)

        # Valid indices: need [i-T+1 .. i] and target [i+h]
        self.i0 = T - 1
        self.i1 = len(seq) - horizon - 1

        # Precompute valid indices using stride
        self.indices = list(range(self.i0, self.i1 + 1, self.stride))        

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        i = self.indices[k]

        x = self.seq[i - self.T + 1 : i + 1]     # (T, 2H, W)
        y = self.seq[i + self.h]                 # (2H, W)

        H = self.n_subap
        xX = x[:, :H, :]
        xY = x[:, H:, :]
        x = torch.stack([xX, xY], dim=1)         # (T,2,H,W)

        yX = y[:H, :]
        yY = y[H:, :]
        y = torch.stack([yX, yY], dim=0)         # (2,H,W)

        if self.mask2 is not None:
            x = x * self.mask2.unsqueeze(0)
            y = y * self.mask2

        return x, y

def masked_mse(pred, target, mask2, eps=1e-8):
    """
    pred/target: (B,2,H,W)
    mask2: (2,H,W) with 0/1
    returns: scalar mean squared error over valid elements
    """
    m = mask2.to(pred.device).unsqueeze(0)       # (1,2,H,W)
    diff2 = (pred - target) ** 2
    num = (diff2 * m).sum()
    den = (m.sum() * pred.size(0)) + eps         # B * (#valid elements in 2 channels)
    return num / den

# -----------------------------
# Minimal training loop
# -----------------------------
def train_one_epoch(model, loader, optim, device="cuda", mask2=None):
    model.train()
    total = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device)  # (B, T, 2, H, W)
        y = y.to(device)  # (B, 2, H, W)

        pred = model(x)

        if mask2 is None:
            loss = F.mse_loss(pred, y, reduction="mean")
        else:
            loss = masked_mse(pred, y, mask2)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        bs = x.size(0)
        total += loss.item() * bs
        n += bs

    return total / max(n, 1)


@torch.no_grad()
def eval_one_epoch(model, loader, device="cuda", mask2=None):
    model.eval()
    total = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)

        if mask2 is None:
            loss = F.mse_loss(pred, y, reduction="mean")
        else:
            loss = masked_mse(pred, y, mask2)

        bs = x.size(0)
        total += loss.item() * bs
        n += bs

    return total / max(n, 1)

@torch.no_grad()
def evaluate_sequence(model, seq, nSubap, T, horizon, mask2=None, std=None, device="cuda"):
    """
    seq: (N, 2H, W) torch tensor
    returns:
        preds: list of predicted slopes
        targets: list of true slopes
        mse_mean: scalar
    """
    model.eval()
    seq = seq.to(device)

    H = nSubap
    N = seq.shape[0]

    errors = []
    preds = []
    targets = []

    for t in range(T-1, N-horizon):
        # Build window
        window = seq[t-T+1:t+1]  # (T,2H,W)

        xX = window[:, :H, :]
        xY = window[:, H:, :]
        x = torch.stack([xX, xY], dim=1)  # (T,2,H,W)

        if mask2 is not None:
            x = x * mask2.unsqueeze(0)

        if std is not None:
            x = x / std.view(1,2,1,1)

        x = x.unsqueeze(0)  # (1,T,2,H,W)

        pred = model(x).squeeze(0)  # (2,H,W)

        target = seq[t+horizon]
        yX = target[:H, :]
        yY = target[H:, :]
        target = torch.stack([yX, yY], dim=0)

        if mask2 is not None:
            target = target * mask2

        if std is not None:
            target = target / std

        mse = F.mse_loss(pred, target, reduction="mean")
        errors.append(mse.item())

        preds.append(pred.cpu())
        targets.append(target.cpu())

    mse_mean = sum(errors) / len(errors)
    return preds, targets, mse_mean