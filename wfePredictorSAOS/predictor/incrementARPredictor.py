"""
Autoregressive predictor of the slope increment, with weights shared by all the slopes (temporal only, like the LSTM):

    s_hat(t+h) = s(t) + sum_j w_j * ds(t-j),   ds(t) = s(t) - s(t-1),  j = 0..p-1

The prediction is built on the last sample, so the gain at zero frequency is exactly 1, and it is equivariant in offset and amplitude
(being linear, any per-window normalisation cancels out in the prediction). The normalisation by the window rms of the increments is
used in the fit: every window weighs the same in the least squares, independently of the amplitude (r0, geometry, plate scale).

Fit (CLI) on the v2 dataset (POL slopes as input, open-loop slopes as target; also reads the v1 open-loop files). With --lowpass the cost is the low-pass filtered prediction error
(zero-phase Butterworth): the predictor optimises only the band that the loop can correct instead of the high-frequency staircase of
the atmosphere. The predictor is a linear time-invariant filter, so the filtered error equals the error of the same predictor applied
to the filtered slopes: the fit is done on the filtered slopes (features and target), and the weights are then used on raw slopes.
    python incrementARPredictor.py --p 16 --horizons 1 2 3 4 [--lowpass 100 --hf_weight 0.1]
"""
import os
import glob
import argparse
import h5py
import numpy as np


class IncrementARPredictor:
    def __init__(self, n_slopes, weights_file, horizon=2):
        d = np.load(weights_file)
        horizons = list(d['horizons'])
        if horizon not in horizons:
            raise ValueError(f'IncrementARPredictor - horizon {horizon} not fitted in {weights_file} ({horizons})')
        self.w = d['W'][horizons.index(horizon)].astype(np.float64)        # (p,), w[j] multiplies ds(t-j)
        self.p = len(self.w)
        self.n_slopes = n_slopes
        self.buffer = np.zeros((self.p + 1, n_slopes))
        self.n_pushed = 0

    def push(self, slopes):
        slopes = np.asarray(slopes, dtype=np.float64)
        if slopes.shape != (self.n_slopes,):
            raise ValueError(f'IncrementARPredictor - expected shape {(self.n_slopes,)}, received {slopes.shape}')
        self.buffer = np.roll(self.buffer, -1, axis=0)
        self.buffer[-1] = slopes
        self.n_pushed += 1

    def ready(self):
        return self.n_pushed >= self.p + 1

    def predict(self):
        ds = np.diff(self.buffer, axis=0)[::-1]                             # ds[j] = ds(t-j)
        return self.buffer[-1] + self.w @ ds


def lowpass_zero_phase(S, fc, fs=2000.0, order=4):
    from scipy.signal import butter, sosfiltfilt
    sos = butter(order, fc, fs=fs, output='sos')
    return sosfiltfilt(sos, S, axis=0)


def load_input_target(path):
    """
    (input, target) slope series (T, N) of a data file, aligned by atmosphere instant:
      v2 dataset (generateTrainingData.py): input = POL slopes of the closed loop, target = open-loop slopes of the parallel WFS
      v1 training data (SAOS Savepoint, open loop): input = target = open-loop slopes
    """
    with h5py.File(path, 'r') as h:
        if 'slopes/pol' in h:
            S_in, S_tgt = h['slopes/pol'][:].astype(np.float64), h['slopes/ol'][:].astype(np.float64)
            ok = np.isfinite(S_in).all(axis=1) & np.isfinite(S_tgt).all(axis=1)
            return S_in[ok], S_tgt[ok]
        S = h['LightPath_0/slopes_1D/data'][:].squeeze().astype(np.float64)
        return S, S


def fit_increment_ar(files, p=16, horizons=(2, 3), sigma_window=24, sigma_floor_frac=0.1, lowpass=None, hf_weight=0.0, fs=2000.0):
    """
    Weighted least squares on all the slopes and frames of the files: target (s_tgt(t+h) - s_in(t)) / sigma, features ds_in(t-j) / sigma,
    with sigma the rms of the raw input increments of the last sigma_window frames of that slope (floor: sigma_floor_frac times the median
    of the file). With lowpass [Hz] the cost is |LP(e)|^2 + hf_weight*|e|^2, with e the prediction error and LP a zero-phase low-pass
    filter: the first term is computed on the filtered series (features and target), the second on the raw ones, both with the raw sigma.
    hf_weight bounds the high-frequency gain of the predictor (with 0 it is unconstrained above the cut-off).
    """
    hmax = max(horizons)
    A = np.zeros((p, p))
    B = np.zeros((len(horizons), p))
    n_rows = 0
    for f in files:
        S_in, S_tgt = load_input_target(f)
        dS_raw = np.diff(S_in, axis=0, prepend=S_in[:1])                               # dS[t] = s(t) - s(t-1)
        # Window rms of the raw input increments ending at t
        c = np.cumsum(np.vstack([np.zeros((1, S_in.shape[1])), dS_raw ** 2]), axis=0)
        t = np.arange(max(p, sigma_window), S_in.shape[0] - hmax)
        sig = np.sqrt((c[t + 1] - c[t + 1 - sigma_window]) / sigma_window)
        sig = np.maximum(sig, sigma_floor_frac * np.median(sig))
        terms = [(S_in, S_tgt, 1.0)]
        if lowpass is not None:
            terms = [(lowpass_zero_phase(S_in, lowpass, fs), lowpass_zero_phase(S_tgt, lowpass, fs), 1.0)] + ([(S_in, S_tgt, hf_weight)] if hf_weight > 0 else [])
        for Si, St, wt in terms:
            dS = np.diff(Si, axis=0, prepend=Si[:1])
            X = (np.stack([dS[t - j] for j in range(p)], axis=-1) / sig[..., None]).reshape(-1, p)
            A += wt * (X.T @ X)
            for ih, hh in enumerate(horizons):
                B[ih] += wt * (X.T @ ((St[t + hh] - Si[t]) / sig).reshape(-1))
        n_rows += t.size * S_in.shape[1]
    return np.stack([np.linalg.solve(A + 1e-9 * np.trace(A) / p * np.eye(p), B[ih]) for ih in range(len(horizons))]), n_rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fit the increment AR predictor (v2 dataset: POL input, open-loop target)')
    parser.add_argument('--files', default=os.path.join(os.path.expanduser('~'), 'simulations', 'results', 'wfePredictor', 'v2', 'dataset', 'train', '*.h5'),
                        help='Glob of the data files (default: train split of the v2 dataset)')
    parser.add_argument('--p', type=int, default=16)
    parser.add_argument('--horizons', type=int, nargs='+', default=[1, 2, 3, 4])
    parser.add_argument('--lowpass', type=float, default=None, help='Cut-off [Hz] of the low-pass weighting of the prediction error (default: none)')
    parser.add_argument('--hf_weight', type=float, default=0.0, help='Weight of the raw (unfiltered) prediction error with --lowpass (default 0)')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    files = sorted(glob.glob(args.files))
    if not files:
        raise SystemExit(f'No files match {args.files}')
    print(f'Fitting p={args.p}, horizons {args.horizons}, low-pass {args.lowpass}, hf weight {args.hf_weight} on {len(files)} files')
    W, n = fit_increment_ar(files, p=args.p, horizons=args.horizons, lowpass=args.lowpass, hf_weight=args.hf_weight)
    tag = f'incAR_p{args.p}' + (f'_lp{args.lowpass:g}' if args.lowpass else '') + (f'_hf{args.hf_weight:g}' if args.lowpass and args.hf_weight else '')
    out = args.out or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'weights', f'{tag}.npz')
    np.savez(out, p=args.p, horizons=np.array(args.horizons), W=W, lowpass=args.lowpass or 0, hf_weight=args.hf_weight, files=np.array([os.path.basename(f) for f in files]))
    for hh, w in zip(args.horizons, W):
        print(f'  h={hh}: w = {np.round(w, 3)}')
    print(f'Saved {out} ({n:,} rows)')
