import os
import glob
import argparse
import numpy as np
import h5py
import torch
import torch.nn.functional as F


class StencilPredictor:
    def __init__(self,
                 valid_subapertures,
                 K=3,
                 past_horizon=8,
                 horizon=3,
                 mode='rls',
                 lam=0.995,
                 weights_file=None,
                 weights=None,
                 init='global',
                 device=None):
        """
        Local spatio-temporal linear predictor of the WFS slopes (stencil).

        The increment of every slope over `horizon` samples is predicted as a linear combination of the x and y slopes of the K x K
        neighbouring subapertures over the last `past_horizon` frames. The same weights are used at every subaperture (a convolution
        over the subaperture grid), so the predictor does not depend on the WFS geometry. Missing neighbours (pupil border, central
        obstruction) are zero.

            s_a(r, c, t+h) = s_a(r, c, t) + sum_{k, b, i, j} w[k, b, i, j, a] * s_b(r+i, c+j, t-k)

        Modes:
            'global': fixed weights fitted offline (see fit_global_stencil).
            'rls'   : weights updated online with exponentially weighted least squares (forgetting factor lam). The weights for horizon h
                      are updated with the features of t-h once s(t) is pushed, so only past data are used. The ridge prior pulls the
                      weights towards the global ones (init='global') or towards zero, i.e. ZOH (init='zero'), with the weight of one frame.

        Parameters
        ----------
        valid_subapertures : np.ndarray (N, N) bool
            Valid subapertures of the WFS. The 1D slopes are [x of the valid subapertures (row-major), y of the valid subapertures].
        K : int
            Neighbourhood size (odd).
        past_horizon : int
            Number of past frames used (p).
        horizon : int
            Prediction horizon in samples from the last pushed frame.
        mode : str
            'global' or 'rls'.
        lam : float
            Forgetting factor of the RLS (memory ~ 1/(1-lam) frames).
        weights_file : str, optional
            npz with the global weights (default: stencil_global_K{K}_p{p}.npz next to this file). Used for mode='global' or init='global'.
        weights : np.ndarray (p*2K^2, 2), optional
            Global weights given directly for `horizon` (instead of weights_file).
        init : str
            'global' or 'zero', RLS prior.
        device : str, optional
            Torch device.
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.mask = torch.as_tensor(np.asarray(valid_subapertures, dtype=bool), device=self.device)
        self.mask_flat = self.mask.flatten()
        self.n_valid = int(self.mask.sum())
        self.n_slopes = 2 * self.n_valid
        self.K, self.p, self.h = int(K), int(past_horizon), int(horizon)
        self.mode, self.lam = mode, float(lam)
        self.n_feat = self.p * 2 * self.K * self.K

        if mode not in ('global', 'rls'):
            raise ValueError("StencilPredictor::__init__ - mode must be 'global' or 'rls'.")

        self.w0 = torch.zeros((self.n_feat, 2), dtype=torch.float64, device=self.device)
        if weights is not None:
            self.w0 = torch.as_tensor(weights, dtype=torch.float64, device=self.device)
        elif mode == 'global' or init == 'global':
            if weights_file is None:
                weights_file = os.path.join(os.path.dirname(__file__), f'stencil_global_K{self.K}_p{self.p}.npz')
            w = np.load(weights_file)
            if int(w['K']) != self.K or int(w['p']) != self.p:
                raise ValueError(f'StencilPredictor::__init__ - {weights_file} has K={int(w["K"])}, p={int(w["p"])}.')
            horizons = list(w['horizons'])
            if self.h not in horizons:
                raise ValueError(f'StencilPredictor::__init__ - horizon {self.h} not fitted in {weights_file} (horizons {horizons}).')
            self.w0 = torch.as_tensor(w['W'][horizons.index(self.h)], dtype=torch.float64, device=self.device)
        self.W = self.w0.clone()

        # Ring buffers (oldest first): unfolded neighbourhoods (2K^2, n_valid) and slopes (2, n_valid)
        self.U = []
        self.S = []
        self.X_hist = []                # features of the last h+1 frames, for the delayed RLS update
        self.R = torch.zeros((self.n_feat, self.n_feat), dtype=torch.float64, device=self.device)
        self.b = torch.zeros((self.n_feat, 2), dtype=torch.float64, device=self.device)
        self.delta = None

    def _to_grid(self, slopes):
        s = torch.as_tensor(np.asarray(slopes), dtype=torch.float64, device=self.device).reshape(-1)
        if s.shape[0] != self.n_slopes:
            raise ValueError(f'StencilPredictor::push - expected {self.n_slopes} slopes, received {s.shape[0]}.')
        G = torch.zeros((2,) + tuple(self.mask.shape), dtype=torch.float64, device=self.device)
        G[0][self.mask] = s[:self.n_valid]
        G[1][self.mask] = s[self.n_valid:]
        return G, torch.stack([s[:self.n_valid], s[self.n_valid:]])

    def push(self, slopes):
        G, s = self._to_grid(slopes)
        U = F.unfold(G.unsqueeze(0), self.K, padding=self.K // 2)[0][:, self.mask_flat]       # (2K^2, n_valid)
        self.U.append(U)
        self.S.append(s)
        if len(self.U) > self.p:
            self.U.pop(0)
        if len(self.S) > self.h + 1:
            self.S.pop(0)
        if not self.ready():
            return
        # Features of this frame: rows = subapertures, columns ordered (lag k, channel b, i, j) with lag 0 the newest frame
        X = torch.cat([self.U[-1 - k].T for k in range(self.p)], dim=1)                     # (n_valid, p*2K^2)
        self.X_hist.append(X)
        if len(self.X_hist) > self.h + 1:
            self.X_hist.pop(0)
        if self.mode == 'rls':
            if self.delta is None:
                self.delta = torch.trace(X.T @ X) / self.n_feat
            if len(self.X_hist) == self.h + 1 and len(self.S) == self.h + 1:
                Xo = self.X_hist[0]
                target = (self.S[-1] - self.S[0]).T                                          # (n_valid, 2): s(t) - s(t-h)
                self.R = self.lam * self.R + Xo.T @ Xo
                self.b = self.lam * self.b + Xo.T @ target
                eye = torch.eye(self.n_feat, dtype=torch.float64, device=self.device)
                self.W = torch.linalg.solve(self.R + self.delta * eye, self.b + self.delta * self.w0)

    def ready(self):
        return len(self.U) == self.p

    def predict(self):
        """Prediction of the slopes `horizon` samples after the last pushed frame (1D, same order as the input)."""
        if not self.ready():
            raise RuntimeError('StencilPredictor::predict - The buffer is not full yet.')
        incr = self.X_hist[-1] @ self.W                                                       # (n_valid, 2)
        pred = self.S[-1].T + incr
        return torch.cat([pred[:, 0], pred[:, 1]]).cpu().numpy()


def fit_global_stencil(files, K=3, past_horizon=8, horizons=(2, 3, 4), device=None):
    """
    Least-squares fit of the global stencil weights on open-loop slopes (SAOS Savepoint files with LightPath_0/slopes_1D and slopes_2D).
    The valid subapertures are taken from the non-zero entries of slopes_2D. All horizons are fitted on the same frames.
    Returns W (n_horizons, p*2K^2, 2).
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    p, hmax = past_horizon, max(horizons)
    n_feat = p * 2 * K * K
    A = torch.zeros((n_feat, n_feat), dtype=torch.float64, device=device)
    B = torch.zeros((len(horizons), n_feat, 2), dtype=torch.float64, device=device)
    for f in files:
        with h5py.File(f, 'r') as h:
            S = h['LightPath_0/slopes_1D/data'][:].squeeze()
            g2 = h['LightPath_0/slopes_2D/data'][:, 0]
        n = g2.shape[-1]
        mask = np.abs(g2[:, :n]).sum(0) > 0
        if 2 * mask.sum() != S.shape[1]:
            raise ValueError(f'fit_global_stencil - inconsistent valid subapertures in {f}.')
        # Fixed zero weights: the predictor is only used to build the features
        pred = StencilPredictor(mask, K=K, past_horizon=p, horizon=hmax, mode='global', weights=np.zeros((n_feat, 2)), device=device)
        frames = []                                                                          # (X, s) of the last hmax+1 frames
        for t in range(S.shape[0]):
            pred.push(S[t])
            frames.append((pred.X_hist[-1] if pred.ready() else None, pred.S[-1]))
            if len(frames) > hmax + 1:
                frames.pop(0)
            if len(frames) == hmax + 1 and frames[0][0] is not None:
                X0, s0 = frames[0]
                A += X0.T @ X0
                for ih, hh in enumerate(horizons):
                    B[ih] += X0.T @ (frames[hh][1] - s0).T
    eye = 1e-9 * torch.trace(A) / n_feat * torch.eye(n_feat, dtype=torch.float64, device=device)
    return np.stack([torch.linalg.solve(A + eye, B[ih]).cpu().numpy() for ih in range(len(horizons))])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fit the global stencil weights on the open-loop training data of the LSTM (atm1-4)')
    parser.add_argument('--data_dir', default='/home/nlinares/simulations/results/predictorSAOS/training/training_val')
    parser.add_argument('--atms', type=int, nargs='+', default=[1, 2, 3, 4])
    parser.add_argument('--K', type=int, default=3)
    parser.add_argument('--p', type=int, default=8)
    parser.add_argument('--horizons', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    files = sorted(f for a in args.atms for f in glob.glob(os.path.join(args.data_dir, f'res_atm{a}_*.h5')))
    print(f'Fitting K={args.K}, p={args.p}, horizons {args.horizons} on {len(files)} files')
    W = fit_global_stencil(files, K=args.K, past_horizon=args.p, horizons=args.horizons)
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), f'stencil_global_K{args.K}_p{args.p}.npz')
    np.savez(out, K=args.K, p=args.p, horizons=np.array(args.horizons), W=W, files=np.array([os.path.basename(f) for f in files]))
    print(f'Saved {out}')
