"""
Dataset v2 of the predictor: POL slopes of a closed loop (input of the predictor, as in operation) and the true open-loop slopes of the
same atmosphere (target), measured with the same WFS object by two light paths propagated sequentially:
  CL  closed loop with the POL law without predictor (gain drawn per run), LightPath delay 1 -> s_pol(i-1)
  OL  open loop (no DM), no delay -> s_OL(i)
Both series are stored aligned by atmosphere instant t (slopes/pol[t], slopes/ol[t]); the last `delay` POL instants are NaN.

Every run draws its own conditions with the seed seed_base(split) + index:
  LOS r0 log-uniform in [5, 21] cm, zenith uniform [10, 60] deg, L0 uniform [15, 50] m, 5 layers at the fixed altitudes with a
  Dirichlet fractional profile (ground dominated on average), wind speed per layer uniform in the ranges of WIND_RANGES and uniform
  directions, POL gain uniform in GAIN_RANGES[sensor]. The phase screens of each run are generated with the same seed and removed after
  the run (unless --keep_screens).

Splits: train / val / test (different seed bases). The 36x36 runs are used for training and validation; the 50x50 test runs check the
transfer to another geometry (never used in training).

Output: <base_dir>/results/predictor_dataset_v2/<split>/run_<sensor>x<sensor>_<split>_<index>.h5

Usage:
  python generateTrainingData.py --split train --index 0
  python generateTrainingData.py --split test --sensor 50 --index 3
  python generateTrainingData.py --test                  (60 iterations, results in predictor_dataset_v2_test)
"""
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import sys
import time
import json
import argparse
import logging
import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from SAOS.LoggingHelper import LoggingHelper
from SAOS.LightPath import LightPath

from common import get_base_dir, get_asset_dirs, make_telescope, build_optics, load_atmosphere, PolControl

SEED_BASE = {'train': 10000, 'val': 20000, 'test': 30000}
ALTITUDES = [100, 1500, 5000, 10000, 15000]
WIND_RANGES = [(2, 12), (3, 15), (5, 25), (10, 35), (3, 15)]       # m/s per layer
GAIN_RANGES = {36: (0.3, 1.0), 50: (0.3, 0.6)}                       # POL g 1.0 is unstable on 50x50 (M7)


def draw_conditions(seed, sensor):
    rng = np.random.default_rng(seed)
    los_r0 = float(np.exp(rng.uniform(np.log(0.05), np.log(0.21))))
    zenith = float(rng.uniform(10, 60))
    frac = rng.dirichlet([4.0, 2.0, 1.2, 0.8, 0.5])
    atm = {'r0': los_r0 / np.cos(np.deg2rad(zenith)) ** (3 / 5), 'L0': float(rng.uniform(15, 50)), 'zenith': zenith,
           'fractionalR0': [float(x) for x in frac], 'altitude': ALTITUDES,
           'windSpeed': [float(rng.uniform(a, b)) for a, b in WIND_RANGES], 'windDirection': [float(x) for x in rng.uniform(0, 360, 5)]}
    gain = float(rng.uniform(*GAIN_RANGES[sensor]))
    return atm, los_r0, gain


def parse_args():
    parser = argparse.ArgumentParser(description="Dataset v2 of the predictor: POL slopes (closed loop) and open-loop slopes of the same atmosphere")
    parser.add_argument('--split', choices=list(SEED_BASE), default='train')
    parser.add_argument('--index', type=int, nargs='+', default=[0], help="Run indices (one seed each)")
    parser.add_argument('--sensor', type=int, default=36, choices=[36, 50])
    parser.add_argument('--n_iterations', type=int, default=2000)
    parser.add_argument('--delay', type=int, default=1)
    parser.add_argument('--base_dir', type=str, default=None)
    parser.add_argument('--keep_screens', action='store_true')
    parser.add_argument('--skip_existing', action='store_true')
    parser.add_argument('--test', action='store_true', help="60 iterations, results in predictor_dataset_v2_test")
    return parser.parse_args()


def run(args, index, base_dir, logger):
    sensor, d = args.sensor, args.delay
    seed = SEED_BASE[args.split] + 1000 * (sensor == 50) + index
    out_dir = os.path.join(base_dir, 'results', 'predictor_dataset_v2' + ('_test' if args.test else ''), args.split)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f'run_{sensor}x{sensor}_{args.split}_{index:03d}.h5')
    if args.skip_existing and os.path.exists(out):
        logger.info(f'Skipping {os.path.basename(out)}: already exists')
        return
    atm_kw, los_r0, gain = draw_conditions(seed, sensor)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    t0 = time.time()
    mirror_models_dir, _ = get_asset_dirs(base_dir)
    tel = make_telescope(logger)
    ps_path = os.path.join(base_dir, 'phase_screens', 'predictor_v2', f'ps_v2_{args.split}_{sensor}_{index:03d}_seed{seed}.h5')
    atm = load_atmosphere(tel, atm_kw, ps_path, seed, logger)
    sun, _, dm, make_wfs = build_optics(sensor, tel, mirror_models_dir, logger)
    wfs = make_wfs()                                   # shared by both light paths
    lp_cl, lp_ol = LightPath(logger), LightPath(logger)
    lp_cl.initialize_path(src=sun, atm=atm, tel=tel, dm=[dm], wfs=wfs, delay=d)
    lp_ol.initialize_path(src=sun, atm=atm, tel=tel, dm=None, wfs=wfs, delay=0)
    ctrl = PolControl(base_dir, sensor, [lp_cl], dm, d, tel, logger, device=device)
    n_slopes = lp_cl.slopes_1D.shape[0]

    T = args.n_iterations
    pol = np.full((T, n_slopes), np.nan, dtype=np.float32)
    ol = np.full((T, n_slopes), np.nan, dtype=np.float32)
    res_rms = np.full(T, np.nan)
    diverged = False
    logger.info(f'>>> {os.path.basename(out)} | seed {seed} | LOS r0 {100 * los_r0:.1f} cm, zenith {atm_kw["zenith"]:.0f}, '
                f'winds {np.round(atm_kw["windSpeed"], 1).tolist()} m/s, fractions {np.round(atm_kw["fractionalR0"], 2).tolist()} | POL g {gain:.2f}')
    t_loop = time.time()
    for i in range(T):
        atm.update()
        lp_cl.propagate(True)
        lp_ol.propagate(True)
        ol[i] = lp_ol.slopes_1D
        s_res = ctrl.tensor(lp_cl.get_wavefront_error())
        s_pol = ctrl.pol_slopes(s_res)
        if i - d >= 0:
            pol[i - d] = s_pol.squeeze(1).cpu().numpy()
        ctrl.step_pol(s_pol, gain)
        res_rms[i] = float(s_res.std())
        if not np.isfinite(res_rms[i]) or res_rms[i] > 5.0:
            logger.warning(f'  loop diverged at iteration {i} (residual slopes rms {res_rms[i]:.3e} px)')
            diverged = True
            break
        if (i + 1) % 500 == 0 or i == 0:
            logger.info(f'  [it {i + 1:5d}/{T}] residual slopes rms {res_rms[i]:.4f} px | {(time.time() - t_loop) / (i + 1):.3f} s/it')

    with h5py.File(out, 'w') as f:
        opts = dict(compression='gzip', compression_opts=4)
        f.create_dataset('slopes/pol', data=pol, **opts)
        f.create_dataset('slopes/ol', data=ol, **opts)
        f.create_dataset('rms/residual_slopes', data=res_rms)
        f.create_dataset('valid_subapertures', data=np.asarray(wfs.valid_subapertures, dtype=bool))
        cfg = dict(split=args.split, index=index, seed=seed, sensor=sensor, delay=d, gain=gain, decay=ctrl.decay, los_r0=los_r0, atmosphere=atm_kw,
                   n_iterations=T, diverged=diverged)
        f.attrs['config'] = json.dumps(cfg)
        f.attrs['description'] = __doc__
    if not args.keep_screens and os.path.exists(ps_path):
        os.remove(ps_path)
    ok = np.isfinite(pol).all(1)
    logger.info(f'<<< {os.path.basename(out)}: OL rms {np.sqrt(np.nanmean(ol ** 2)):.3f} px, POL error {np.sqrt(np.mean((pol[ok] - ol[ok]) ** 2)):.4f} px, '
                f'residual {np.sqrt(np.nanmean(res_rms[T // 5:] ** 2)):.4f} px{" DIVERGED" if diverged else ""} | {time.time() - t0:.0f} s')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    if args.test:
        args.n_iterations = 60
    log_helper = LoggingHelper(logging.INFO)          # keep the reference: the queue listener stops when it is deleted
    logger = log_helper.logger
    base_dir = get_base_dir(args.base_dir)
    for index in args.index:
        run(args, index, base_dir, logger)
    log_helper = None


if __name__ == '__main__':
    main()
