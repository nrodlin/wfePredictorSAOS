import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

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
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.0, gn_groups=8, use_wn=True):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        Conv = nn.Conv1d
        self.conv1 = Conv(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation, bias=False)
        self.conv2 = Conv(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation, bias=False)
        if use_wn:
            self.conv1 = weight_norm(self.conv1)
            self.conv2 = weight_norm(self.conv2)

        def GN(c):
            g = min(gn_groups, c)
            while c % g != 0 and g > 1:
                g -= 1
            return nn.GroupNorm(g, c)

        self.chomp1 = Chomp1d(padding)
        self.gn1 = GN(out_ch)

        self.chomp2 = Chomp1d(padding)
        self.gn2 = GN(out_ch)

        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else None

    def forward(self, x):
        y = self.conv1(x)
        y = self.chomp1(y)
        y = self.gn1(y)
        y = F.gelu(y)
        y = self.dropout(y)

        y = self.conv2(y)
        y = self.chomp2(y)
        y = self.gn2(y)
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
    def __init__(self, in_ch=2, base=32, emb=256, gn_groups=8):
        super().__init__()
        # Ajusta grupos para que divida a los canales
        def GN(c):
            g = min(gn_groups, c)
            while c % g != 0 and g > 1:
                g -= 1
            return nn.GroupNorm(g, c)

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1, bias=False),
            GN(base),
            nn.GELU(),

            nn.Conv2d(base, base, 3, padding=1, bias=False),
            GN(base),
            nn.GELU(),

            nn.Conv2d(base, 2*base, 3, stride=2, padding=1, bias=False),
            GN(2*base),
            nn.GELU(),

            nn.Conv2d(2*base, 2*base, 3, padding=1, bias=False),
            GN(2*base),
            nn.GELU(),

            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(2*base, emb)

    def forward(self, x):
        h = self.net(x).squeeze(-1).squeeze(-1)  # (B, 2*base)
        return self.proj(h)                      # (B, emb)


# -----------------------------
# Full model: CNN + TCN + head
# -----------------------------
class SHWFS_TCN(nn.Module):
    def __init__(self, n_subap, emb=256, tcn_layers=4, tcn_kernel=3, dropout=0.0, in_norm=True):
        super().__init__()
        self.n_subap = n_subap

        self.in_norm = nn.GroupNorm(2, 2) if in_norm else None  # normaliza (B,2,H,W) por canal

        self.spatial = SpatialEncoder(in_ch=2, base=16, emb=emb)

        channels = [emb] + [emb] * tcn_layers
        self.tcn = TemporalConvNet(channels, kernel_size=tcn_kernel, dropout=dropout)

        self.head = nn.Sequential(
            nn.Linear(emb, emb),
            nn.GELU(),
            nn.Linear(emb, 2 * n_subap * n_subap),
        )

    def forward(self, x):
        if x.dim() == 4:
            B, T, HH, W = x.shape
            H = HH // 2
            xX = x[:, :, :H, :]
            xY = x[:, :, H:, :]
            x = torch.stack([xX, xY], dim=2)  # (B,T,2,H,W)

        B, T, C, H, W = x.shape
        assert H == self.n_subap and W == self.n_subap and C == 2

        # input norm por frame dentro del modelo
        if self.in_norm is not None:
            x = x.reshape(B*T, C, H, W)
            x = self.in_norm(x)
            x = x.view(B, T, C, H, W)

        # (opcional) vectorización del encoder espacial para quitar el loop:
        x_bt = x.reshape(B*T, C, H, W)
        f_bt = self.spatial(x_bt)  # (B*T, emb)
        feats = f_bt.view(B, T, -1).transpose(1, 2)  # (B, emb, T)

        z = self.tcn(feats)          # (B, emb, T)
        z_last = z[:, :, -1]         # (B, emb)
        y = self.head(z_last).view(B, 2, H, W)
        return y

# -----------------------------
# Example windowed dataset
# -----------------------------
class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, seq, n_subap, T=8, horizon=2, stride=1):
        """
        seq: torch tensor (N, 2*nSubap, nSubap) with N consecutive frames
        """
        super().__init__()
        self.seq = seq
        self.T = T
        self.h = horizon
        self.n_subap = n_subap
        self.stride = stride

        self.i0 = T - 1
        self.i1 = len(seq) - horizon - 1
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
def evaluate_sequence(model, seq, nSubap, T, horizon, mask2=None, device="cuda"):
    """
    seq: (N, 2H, W) torch tensor
    """
    model.eval()
    seq = seq.to(device)

    H = nSubap
    N = seq.shape[0]

    errors = []
    preds = []
    targets = []

    for t in range(T-1, N-horizon):
        window = seq[t-T+1:t+1]  # (T,2H,W)

        xX = window[:, :H, :]
        xY = window[:, H:, :]
        x = torch.stack([xX, xY], dim=1)  # (T,2,H,W)
        x = x.unsqueeze(0)                # (1,T,2,H,W)

        pred = model(x).squeeze(0)        # (2,H,W)

        target = seq[t+horizon]
        yX = target[:H, :]
        yY = target[H:, :]
        target = torch.stack([yX, yY], dim=0)  # (2,H,W)

        if mask2 is None:
            mse = F.mse_loss(pred, target, reduction="mean").item()
        else:
            mse = masked_mse(pred.unsqueeze(0), target.unsqueeze(0), mask2).item()

        errors.append(mse)
        preds.append(pred.cpu())
        targets.append(target.cpu())

    mse_mean = sum(errors) / max(len(errors), 1)
    return preds, targets, mse_mean

@torch.no_grad()
def eval_model_and_baseline(model, seq, nSubap, T=8, horizon=2, mask2=None, device="cuda"):
    """
    seq: (N, 2H, W)
    mask2: (2, H, W) float {0,1}
    baseline: persistencia (frame t)
    """
    model.eval()
    seq = seq.to(device)
    H = nSubap
    N = seq.shape[0]

    e_model, e_base = [], []

    for t in range(T-1, N-horizon):
        window = seq[t-T+1:t+1]          # (T,2H,W)
        target2H = seq[t+horizon]        # (2H,W)

        target = torch.stack([target2H[:H, :], target2H[H:, :]], dim=0)  # (2,H,W)
        base2H = seq[t]
        base = torch.stack([base2H[:H, :], base2H[H:, :]], dim=0)        # (2,H,W)

        # modelo
        xX = window[:, :H, :]
        xY = window[:, H:, :]
        x = torch.stack([xX, xY], dim=1).unsqueeze(0)  # (1,T,2,H,W)
        pred = model(x).squeeze(0)                     # (2,H,W)

        if mask2 is None:
            em = torch.mean((pred - target) ** 2).item()
            eb = torch.mean((base - target) ** 2).item()
        else:
            # masked mse inline (evita depender de otra función)
            m = mask2.to(device).unsqueeze(0)  # (1,2,H,W)
            em = (((pred.unsqueeze(0) - target.unsqueeze(0)) ** 2) * m).sum().item() / (m.sum().item() + 1e-12)
            eb = (((base.unsqueeze(0) - target.unsqueeze(0)) ** 2) * m).sum().item() / (m.sum().item() + 1e-12)

        e_model.append(em)
        e_base.append(eb)

    m_model = sum(e_model) / max(len(e_model), 1)
    m_base  = sum(e_base) / max(len(e_base), 1)
    return m_model, m_base, e_model, e_base