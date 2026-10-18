"""
Training of the LSTM predictor v2 (predictor/incrementalLSTM.py) on the v2 dataset (simulations/generateTrainingData.py):
input = POL slopes of the closed loop, target = true open-loop slopes h samples later, h in --horizons.

Every batch is a time segment of a random subset of slopes of one run: the model predicts every instant of the segment, so the prediction
error is a time series e_h(t) per slope and horizon, normalised by the window scale sigma(t) (amplitude independent). Losses:
  band : mean(LP(e)^2) + hf_weight * mean(e^2), LP a zero-phase FIR low-pass at --lowpass Hz: the error that the loop can correct
         (below its bandwidth) dominates; hf_weight bounds the high-frequency gain (the high-frequency staircase of the atmosphere
         motion cannot be corrected by the loop and dominates a plain MSE)
  mse  : mean(e^2)
Validation: the same loss and the skill vs ZOH (prediction = last POL sample) in px, total and low-pass, per horizon, on fixed segments
of the val split. Early stopping on the validation loss. The best model is saved in weights/lstm_v2[_tag].pt with its configuration.

Usage:
  python trainIncrementalLSTM.py --loss band --lowpass 100 --hf_weight 0.1
  python trainIncrementalLSTM.py --loss mse --tag mse
"""
import os
import glob
import json
import time
import argparse
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import firwin

from wfePredictorSAOS.predictor.incrementalLSTM import IncrementalSlopeLSTM

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'weights')
DATA_DIR = os.path.join(os.path.expanduser('~'), 'simulations', 'results', 'wfePredictor', 'v2', 'dataset')


def parse_args():
    parser = argparse.ArgumentParser(description='Training of the incremental LSTM predictor (v2)')
    parser.add_argument('--data_dir', default=DATA_DIR)
    parser.add_argument('--sensor', type=int, default=36, help='Sensor of the train/val runs (default 36)')
    parser.add_argument('--past', type=int, default=24)
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--horizons', type=int, nargs='+', default=[1, 2, 3, 4])
    parser.add_argument('--sigma_floor', type=float, default=1e-3, help='Floor of the window scale [px]')
    parser.add_argument('--loss', choices=['band', 'mse'], default='band')
    parser.add_argument('--lowpass', type=float, default=100.0, help='Cut-off of the band loss [Hz]')
    parser.add_argument('--hf_weight', type=float, default=0.1, help='Weight of the raw error in the band loss')
    parser.add_argument('--fir_taps', type=int, default=101)
    parser.add_argument('--seg_len', type=int, default=256, help='Predicted instants per segment')
    parser.add_argument('--n_slopes', type=int, default=128, help='Slopes per segment')
    parser.add_argument('--batches_per_epoch', type=int, default=200)
    parser.add_argument('--val_segments', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--tag', default='', help='Suffix of the weights file: lstm_v2_<tag>.pt')
    parser.add_argument('--device', default=None)
    parser.add_argument('--max_files', type=int, default=None, help='Use only the first files of each split (tests)')
    return parser.parse_args()


def load_split(data_dir, split, sensor, max_files=None):
    """Runs of a split as (pol, ol) float32 tensors (T, N) without the NaN instants; diverged runs are skipped."""
    runs = []
    for path in sorted(glob.glob(os.path.join(data_dir, split, f'run_{sensor}x{sensor}_{split}_*.h5')))[:max_files]:
        with h5py.File(path, 'r') as f:
            if json.loads(f.attrs['config']).get('diverged', False):
                print(f'  skipping diverged run {os.path.basename(path)}')
                continue
            pol, ol = f['slopes/pol'][:], f['slopes/ol'][:]
        ok = np.isfinite(pol).all(1) & np.isfinite(ol).all(1)
        runs.append((torch.from_numpy(pol[ok]), torch.from_numpy(ol[ok])))
    if not runs:
        raise SystemExit(f'No {split} runs of {sensor}x{sensor} in {data_dir}')
    return runs


class SegmentSampler:
    """Segments (input windows, targets) of seg_len consecutive instants for a subset of slopes of one run."""

    def __init__(self, runs, past, horizons, seg_len, n_slopes):
        self.runs, self.p, self.H, self.L, self.S = runs, past, horizons, seg_len, n_slopes
        self.hmax = max(horizons)

    def sample(self, rng, device):
        pol, ol = self.runs[rng.integers(len(self.runs))]
        T, N = pol.shape
        span = self.p - 1 + self.L + self.hmax
        t0 = int(rng.integers(0, T - span + 1))
        cols = torch.from_numpy(rng.choice(N, size=min(self.S, N), replace=False))
        seg_in = pol[t0:t0 + span][:, cols].to(device)                                  # (span, S)
        seg_ol = ol[t0:t0 + span][:, cols].to(device)
        windows = seg_in.unfold(0, self.p, 1)[:self.L]                                  # (L, S, p): window ending at t0+p-1+l
        x = windows.permute(0, 2, 1).contiguous()                                       # (L, p, S)
        tgt = torch.stack([seg_ol[self.p - 1 + h:self.p - 1 + h + self.L] for h in self.H], dim=-1)   # (L, S, H)
        return x, tgt


def losses(pred, sigma, tgt, x, fir, args):
    """Loss of the model and of ZOH (normalised by sigma), and the raw/low-pass errors in px for the skill."""
    zoh = x[:, -1, :][..., None].expand_as(tgt)
    out = {}
    for name, p in (('model', pred), ('zoh', zoh)):
        e = (p - tgt) / sigma[..., None]                                                # (L, S, H)
        e_t = e.permute(1, 2, 0).reshape(-1, 1, e.shape[0])                             # (S*H, 1, L)
        e_lp = F.conv1d(e_t, fir)                                                       # valid part of the zero-phase FIR
        mse = (e ** 2).mean()
        out[name] = (e_lp ** 2).mean() + args.hf_weight * mse if args.loss == 'band' else mse
        # px errors per horizon, total and low-pass
        r = (p - tgt).permute(1, 2, 0).reshape(-1, 1, e.shape[0])
        out[name + '_px'] = (r ** 2).mean(dim=-1).reshape(-1, len(args.horizons)).mean(0)
        out[name + '_px_lp'] = (F.conv1d(r, fir) ** 2).mean(dim=-1).reshape(-1, len(args.horizons)).mean(0)
    return out


def main():
    args = parse_args()
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    print(f'Loading the dataset from {args.data_dir}')
    train = load_split(args.data_dir, 'train', args.sensor, args.max_files)
    val = load_split(args.data_dir, 'val', args.sensor, args.max_files)
    print(f'  train: {len(train)} runs, val: {len(val)} runs, {train[0][0].shape[1]} slopes')
    fir = torch.as_tensor(firwin(args.fir_taps, args.lowpass, fs=2000.0), dtype=torch.float32, device=device).reshape(1, 1, -1)
    if args.seg_len <= args.fir_taps:
        raise ValueError('seg_len must be larger than fir_taps')

    model = IncrementalSlopeLSTM(past_horizon=args.past, hidden_size=args.hidden, num_layers=args.layers, horizons=args.horizons,
                                 sigma_floor=args.sigma_floor).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
    tr_sampler = SegmentSampler(train, args.past, args.horizons, args.seg_len, args.n_slopes)
    va_sampler = SegmentSampler(val, args.past, args.horizons, args.seg_len, args.n_slopes)
    val_rng = np.random.default_rng(12345)
    val_batches = [va_sampler.sample(val_rng, 'cpu') for _ in range(args.val_segments)]     # fixed validation segments

    name = 'lstm_v2' + (f'_{args.tag}' if args.tag else '')
    out_path = os.path.join(WEIGHTS_DIR, name + '.pt')
    best, best_epoch, history = np.inf, -1, []
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        tr_loss = 0.0
        for _ in range(args.batches_per_epoch):
            x, tgt = tr_sampler.sample(rng, device)
            pred, sigma = model(x)
            loss = losses(pred, sigma, tgt, x, fir, args)['model']
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() / args.batches_per_epoch
        model.eval()
        acc = {k: 0.0 for k in ('model', 'zoh', 'model_px', 'zoh_px', 'model_px_lp', 'zoh_px_lp')}
        with torch.no_grad():
            for x, tgt in val_batches:
                x, tgt = x.to(device), tgt.to(device)
                pred, sigma = model(x)
                o = losses(pred, sigma, tgt, x, fir, args)
                for k in acc:
                    acc[k] = acc[k] + o[k].detach() / len(val_batches)
        va_loss = float(acc['model'])
        skill = (100 * (1 - acc['model_px'] / acc['zoh_px'])).cpu().numpy().astype(np.float64)
        skill_lp = (100 * (1 - acc['model_px_lp'] / acc['zoh_px_lp'])).cpu().numpy().astype(np.float64)
        sched.step(va_loss)
        history.append(dict(epoch=epoch, train=tr_loss, val=va_loss, val_zoh=float(acc['zoh']), skill=skill.tolist(), skill_lp=skill_lp.tolist()))
        improved = va_loss < best
        if improved:
            best, best_epoch = va_loss, epoch
            torch.save({'state_dict': model.state_dict(), 'config': model.config(), 'train_args': vars(args), 'epoch': epoch,
                        'val': history[-1]}, out_path)
        print(f'epoch {epoch:3d} | train {tr_loss:.4f} | val {va_loss:.4f} (ZOH {float(acc["zoh"]):.4f}) | skill vs ZOH [%] h={args.horizons}: '
              f'total {' '.join(f'{v:.1f}' for v in skill)}, <{args.lowpass:g} Hz {' '.join(f'{v:.1f}' for v in skill_lp)} | lr {opt.param_groups[0]["lr"]:.1e} | '
              f'{time.time() - t0:.0f} s{" *" if improved else ""}', flush=True)
        if epoch - best_epoch >= args.patience:
            break
    with open(os.path.join(WEIGHTS_DIR, name + '_history.json'), 'w') as f:
        json.dump(dict(args=vars(args), best_epoch=best_epoch, history=history), f, indent=1)
    print(f'Best epoch {best_epoch}, val loss {best:.4f}: {out_path}')


if __name__ == '__main__':
    main()
