"""
POL reconstruction error and prediction error under the same atmosphere (EST red arm configuration of 02/03, 2 kHz).

Three sensing light paths share exactly the same WFS object (Correlating SH 36x36 or 50x50), propagated sequentially:
  LP1  open loop (no DM), LightPath delay 1  -> at iteration i gives s_OL(i-1): truth for the POL slopes of the same instant
  LP2  closed loop with POL control (no predictor, gain 0.4 by default), LightPath delay 1 (pure delay 1+1 = 2 samples, plus the DM dynamics)
       -> s_pol(i-1) = s_res(i-1) - IM m_DM(i-1)
  LP3  open loop, no delay -> s_OL(i): input of the predictors (ZOH, linear extrapolation, LSTM) that predict s_OL(i+h), and
       truth of their predictions

Errors, all indexed by the atmosphere instant t:
  POL reconstruction error   s_pol(t) - s_OL(t)                                   (LP2 vs LP1)
  prediction error (OL input) p_OL(t) - s_OL(t), with p_OL(t) predicted from s_OL(.. t-h)   (LP3)
  total error (POL input)    p_POL(t) - s_OL(t), with p_POL(t) predicted from s_pol(.. t-h) (the same predictors fed with LP2)
The total error measures directly what the aggregation of the POL and prediction errors should reproduce.
Consistency check: LP1 at t must equal LP3 at t (same WFS, same atmosphere).

Output (results/pol_prediction_errors/res_polpred_..._h5): per-iteration rms of every error and, unless --no-save-slopes, the
slope time series aligned by instant (slopes/ol, slopes/ol_delayed, slopes/pol, pred_ol/<p>, pred_pol/<p>; NaN where not available).

Usage:
  python redArmSolarSCAO_06_POL_prediction_errors.py --sensor 36 --atm atm2 --draw draw1 --no_vibr_only
  python redArmSolarSCAO_06_POL_prediction_errors.py --test          (100 iterations, atm1 draw1 noVibr, results in *_test)
"""
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import sys
import time
import json
import argparse
import logging
import warnings
import h5py
import numpy as np
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from SAOS.LoggingHelper import LoggingHelper
from SAOS.ExtendedSource import ExtendedSource
from SAOS.Telescope import Telescope
from SAOS.Atmosphere import Atmosphere
from SAOS.DeformableMirror import DeformableMirror
from SAOS.Vibration import Vibration
from SAOS.CorrelatingShackHartmann import CorrelatingShackHartmann
from SAOS.LightPath import LightPath
from SAOS.InteractionMatrixHandler import InteractionMatrixHandler
from SAOS.Controller import Controller

from wfePredictorSAOS.predictor.onlinePredictor import OnlineSlopePredictor
from wfePredictorSAOS.predictor.onlineLinearPredictor import OnlineLinearSlopePredictor
from wfePredictorSAOS.predictor.pseudoOpenLoop import PseudoOpenLoop

from redArmSolarSCAO_03_CL_POL import get_base_dir, get_asset_dirs
from atmosphereCases import atm_cases

PREDICTORS = ('zoh', 'linear', 'lstm')

# Profile 1 of the mcaobenchcodes studies (dmRotation SCAO/MCAO6, imageQuality), draw 1
MCAO_PROFILE1 = {'r0': 0.09, 'L0': 25, 'zenith': 15, 'fractionalR0': [0.53, 0.37, 0.05, 0.03, 0.02], 'altitude': [100, 1500, 5000, 10000, 15000],
                 'windSpeed': [6.7, 8.5, 12.7, 23.4, 8.3], 'windDirection': [312, 47, 198, 123, 276]}


def parse_args():
    parser = argparse.ArgumentParser(description="POL reconstruction error and prediction errors under the same atmosphere (3 LPs, same WFS)")
    parser.add_argument('--sensor', type=int, default=36, choices=[36, 50])
    parser.add_argument('--atm', type=str, default=None, help="Atmosphere case (e.g. atm2). Default: all")
    parser.add_argument('--draw', type=str, default=None, help="Draw (e.g. draw1). Default: all")
    parser.add_argument('--mcao_profile1', action='store_true', help="Use profile 1 of mcaobenchcodes (r0 9 cm, zenith 15, draw 1) with its own phase "
                        "screens, generated with --seed the first time and reused afterwards (instead of the validation atmospheres)")
    parser.add_argument('--seed', type=int, default=2026, help="Seed of the phase screens generated for --mcao_profile1 (default 2026)")
    parser.add_argument('--no_vibr_only', action='store_true', help="Only runs without vibrations")
    parser.add_argument('--vibr_only', action='store_true', help="Only runs with vibrations")
    parser.add_argument('--n_iterations', type=int, default=2500, help="Iterations (default 2500 = 1.25 s at 2 kHz)")
    parser.add_argument('--delay', type=int, default=1, help="LightPath delay of LP1 and LP2 (pure delay = delay + 1, default 1)")
    parser.add_argument('--horizon', type=int, default=2, help="Prediction horizon [samples] of every predictor (default 2, the LSTM training)")
    parser.add_argument('--gain', type=float, default=0.4, help="Gain of the POL law of LP2 (default 0.4: g 1.0 diverges on 50x50)")
    parser.add_argument('--decay', type=float, default=0.999)
    parser.add_argument('--beta', type=float, default=5e-4, help="Tikhonov regularisation (default 5e-4, as 03)")
    parser.add_argument('--linear_past', type=int, default=4, help="Past samples of the linear extrapolation (default 4, as in 03)")
    parser.add_argument('--discard', type=float, default=0.2, help="Initial time excluded from the summary [s] (default 0.2)")
    parser.add_argument('--no-save-slopes', dest='save_slopes', action='store_false', help="Do not save the slope time series")
    parser.add_argument('--base_dir', type=str, default=None)
    parser.add_argument('--skip_existing', action='store_true')
    parser.add_argument('--test', action='store_true', help="100 iterations, atm1 draw1 noVibr, results in pol_prediction_errors_test")
    return parser.parse_args()


def rms(x):
    return float(np.sqrt(np.nanmean(x ** 2))) if np.isfinite(x).any() else np.nan


def run_case(args, atm_name, draw_name, kwargs, use_vibrations, base_dir, logger):
    sensor, d, h = args.sensor, args.delay, args.horizon
    vib_label = 'Vibr' if use_vibrations else 'noVibr'
    res_dir = os.path.join(base_dir, 'results', 'pol_prediction_errors' + ('_test' if args.test else ''))
    os.makedirs(res_dir, exist_ok=True)
    res_path = os.path.join(res_dir, f'res_polpred_{sensor}x{sensor}_{vib_label}_{atm_name}_{draw_name}_g{args.gain}_d{d}_h{h}.h5')
    if args.skip_existing and os.path.exists(res_path):
        logger.info(f'Skipping {os.path.basename(res_path)}: already exists')
        return
    mirror_models_dir, vibrations_dir = get_asset_dirs(base_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    t0 = time.time()

    tel = Telescope(diameter=4.149, resolution=200, centralObstruction=1.3 / 4.149, samplingTime=1 / 2000, fov=60.0, logger=logger)
    atm = Atmosphere(r0=kwargs['r0'], L0=kwargs['L0'], fractionalR0=kwargs['fractionalR0'], altitude=kwargs['altitude'],
                     windDirection=kwargs['windDirection'], windSpeed=kwargs['windSpeed'], telescope=tel, zenith=kwargs['zenith'], logger=logger)
    if args.mcao_profile1:
        # Own phase screens of profile 1, generated once with a fixed seed and shared by every sensor/run
        atm_file = os.path.join(base_dir, 'phase_screens', f'ps_polpred_mcaoProfile1_{draw_name}_seed{args.seed}.h5')
        if not os.path.exists(atm_file):
            atm.initializeAtmosphere(randomState=args.seed)
            atm.save(atm_file)
            logger.info(f'Generated phase screens {atm_file} (seed {args.seed})')
        if not atm.load(atm_file):
            raise RuntimeError(f'Could not load {atm_file}')
    else:
        # Same phase screens as the closed-loop campaign (02/03/04): they must exist, a new screen would not be comparable
        atm_file = os.path.join(base_dir, 'phase_screens', f"ps_dualARM_EST_case{int(atm_name.replace('atm', ''))}_{draw_name}.h5")
        if not os.path.exists(atm_file) or not atm.load(atm_file):
            raise FileNotFoundError(f'Phase screens {atm_file} not found (run 02/03 or prepare_assets.py first)')
    vibrations = None
    if use_vibrations:
        atm_idx = int(atm_name.replace('atm', '')) if atm_name.startswith('atm') else 1
        vibrations = Vibration(tel, os.path.join(vibrations_dir, f'EST_vibration_{atm_idx if atm_idx <= 5 else 1}.h5'), logger)

    if sensor == 36:
        sun = ExtendedSource(optBand='R', coordinates=[0, 0], nSubDirs=3, fov=9.269, subDir_margin=4.0, patch_padding=5.0, logger=logger)
        dm = DeformableMirror(telescope=tel, nActs=37, altitude=0, typeDM='radial', logger=logger,
                              dynamicModel=os.path.join(mirror_models_dir, 'asm_discrete_model.h5'), validActThreshpercentage=0.7533)
        wfs_kw = dict(plate_scale=0.403, fieldOfView=9.269)
    else:
        sun = ExtendedSource(optBand='V', coordinates=[0, 0], nSubDirs=3, fov=9.975, subDir_margin=4.0, patch_padding=5.0, logger=logger)
        dm = DeformableMirror(telescope=tel, nActs=51, altitude=0, typeDM='cartesian', logger=logger,
                              dynamicModel=os.path.join(mirror_models_dir, 'm7_discrete_model.h5'), validActThreshpercentage=0.7533)
        wfs_kw = dict(plate_scale=0.475, fieldOfView=9.975)
    # One WFS object shared by the three light paths (each LightPath keeps its own delay buffer and correlation reference)
    wfs = CorrelatingShackHartmann(telescope=tel, src=sun, lightRatio=0.9, nSubap=sensor, guardPx=2, fft_fieldOfView_oversampling=0.5,
                                   use_brightest=9, unit_in_rad=False, logger=logger, **wfs_kw)
    lp1, lp2, lp3 = LightPath(logger), LightPath(logger), LightPath(logger)
    lp1.initialize_path(src=sun, atm=atm, tel=tel, dm=None, wfs=wfs, vibration=vibrations, delay=d)
    lp2.initialize_path(src=sun, atm=atm, tel=tel, dm=[dm], wfs=wfs, vibration=vibrations, delay=d)
    lp3.initialize_path(src=sun, atm=atm, tel=tel, dm=None, wfs=wfs, vibration=vibrations, delay=0)

    im_handler = InteractionMatrixHandler(logger)
    im_handler.initialize_im_class([lp2])
    im_handler.load_modalBasis(os.path.join(base_dir, 'modal_basis', f'predictor_{sensor}x{sensor}_modalBasis.h5'))
    im_handler.load_IM(os.path.join(base_dir, 'interaction_matrix', f'predictor_{sensor}x{sensor}_IM.h5'))
    IM = im_handler.interaction_matrix_warehouse[0][0]['IM']
    controller = Controller(telescope=tel, interactionMatrix=im_handler, reconstructionMethod='tikhonov', controllerType='leaky', logger=logger,
                            rcond=0.025, beta=args.beta, gain=[args.gain], decay=[args.decay], ki=[0.0], nModes=[IM.shape[1]])
    n_modes = controller.reconstructor[0].shape[0]
    offset = controller.discarded_modes[0]
    R = controller.reconstructor[0].to(device)
    M = controller.modal_basis[0][:, offset:offset + n_modes].to(device)
    pol = PseudoOpenLoop(IM, dm, d, n_modes, device=device)
    n_slopes = lp2.slopes_1D.shape[0]

    def make_predictors():
        return {'linear': OnlineLinearSlopePredictor(n_slopes=n_slopes, past_horizon=args.linear_past, steps_ahead=h),
                'lstm': OnlineSlopePredictor(n_slopes=n_slopes, past_horizon=24, hidden_size=16, n_axis=1)}
    pred_ol, pred_pol = make_predictors(), make_predictors()

    T = args.n_iterations
    nan = lambda: np.full((T + h, n_slopes), np.nan, dtype=np.float32)
    ol, ol_delayed, spol = nan(), nan(), nan()
    p_ol = {p: nan() for p in PREDICTORS}
    p_pol = {p: nan() for p in PREDICTORS}
    res_rms = np.full(T, np.nan)
    cmd = torch.zeros((n_modes, 1), dtype=torch.float64, device=device)
    logger.info(f'>>> {os.path.basename(res_path)} | {sensor}x{sensor} {atm_name} {draw_name} {vib_label} | POL g {args.gain}, delay {d}, horizon {h}')
    t_loop = time.time()
    for i in range(T):
        atm.update()
        for lp in (lp1, lp2, lp3):
            lp.propagate(True)
        # LP3: s_OL(i); LP1: s_OL(i-d); LP2: s_res(i-d) -> s_pol(i-d)
        ol[i] = lp3.slopes_1D
        if i - d >= 0:
            ol_delayed[i - d] = lp1.get_wavefront_error()
        s_res = torch.as_tensor(lp2.get_wavefront_error(), dtype=torch.float64, device=device).reshape(-1, 1)
        s_pol = pol.reconstruct(s_res)
        if i - d >= 0:
            spol[i - d] = s_pol.squeeze(1).cpu().numpy()
        # Predictors fed with the open-loop slopes of LP3: prediction of s_OL(i+h)
        p_ol['zoh'][i + h] = ol[i]
        for name, pr in pred_ol.items():
            pr.push(ol[i])
            if pr.ready():
                p_ol[name][i + h] = pr.predict()
        # The same predictors fed with the POL slopes of LP2: prediction of s_OL(i-d+h)
        if i - d >= 0:
            p_pol['zoh'][i - d + h] = spol[i - d]
            for name, pr in pred_pol.items():
                pr.push(spol[i - d])
                if pr.ready():
                    p_pol[name][i - d + h] = pr.predict()
        # LP2 closed loop: POL law without predictor
        target = -(R @ s_pol)
        cmd = args.gain * (target - cmd) + args.decay * cmd
        dm.updateDMShape(M @ cmd)
        pol.update(cmd)
        res_rms[i] = float(s_res.std())
        if (i + 1) % 250 == 0 or i == 0:
            logger.info(f'  [it {i + 1:5d}/{T}] residual slopes rms {res_rms[i]:.4f} px | {(time.time() - t_loop) / (i + 1):.3f} s/it')

    # Errors by instant t (only instants with the truth available)
    ol, ol_delayed, spol = ol[:T], ol_delayed[:T], spol[:T]
    p_ol = {k: v[:T] for k, v in p_ol.items()}
    p_pol = {k: v[:T] for k, v in p_pol.items()}
    err_t = lambda a: np.sqrt(np.nanmean((a - ol) ** 2, axis=1))
    series = {'ol_rms': np.sqrt(np.nanmean(ol ** 2, axis=1)), 'pol_err': err_t(spol), 'lp1_lp3_diff': err_t(ol_delayed)}
    for p in PREDICTORS:
        series[f'pred_err_ol/{p}'] = err_t(p_ol[p])
        series[f'pred_err_pol/{p}'] = err_t(p_pol[p])
    k0 = int(round(args.discard * 2000))
    keep = np.arange(T) >= k0

    def summ(a):
        return rms(a[keep])
    summary = {'ol_rms_px': summ(series['ol_rms']), 'pol_err_px': summ(series['pol_err']), 'lp1_lp3_max_diff_px': float(np.nanmax(series['lp1_lp3_diff'])),
               'residual_slopes_rms_px': rms(res_rms[k0:]), 's_per_it': (time.time() - t_loop) / T, 'wall_s': time.time() - t0}
    for p in PREDICTORS:
        summary[f'pred_err_ol_{p}_px'] = summ(series[f'pred_err_ol/{p}'])
        summary[f'pred_err_pol_{p}_px'] = summ(series[f'pred_err_pol/{p}'])
        # Aggregation hypothesis: independent POL and prediction errors add in quadrature
        summary[f'quadrature_sum_{p}_px'] = float(np.hypot(summary['pol_err_px'], summary[f'pred_err_ol_{p}_px']))
    with h5py.File(res_path, 'w') as f:
        f.create_dataset('instant', data=np.arange(T))
        for k, v in series.items():
            f.create_dataset(f'rms/{k}', data=v)
        f.create_dataset('rms/residual_slopes', data=res_rms)
        if args.save_slopes:
            opts = dict(compression='gzip', compression_opts=4)
            f.create_dataset('slopes/ol', data=ol, **opts)
            f.create_dataset('slopes/ol_delayed', data=ol_delayed, **opts)
            f.create_dataset('slopes/pol', data=spol, **opts)
            for p in PREDICTORS:
                f.create_dataset(f'pred_ol/{p}', data=p_ol[p], **opts)
                f.create_dataset(f'pred_pol/{p}', data=p_pol[p], **opts)
        f.attrs['config'] = json.dumps(dict(sensor=sensor, atm=atm_name, draw=draw_name, vibrations=use_vibrations, gain=args.gain, decay=args.decay,
                                            delay=d, horizon=h, beta=args.beta, linear_past=args.linear_past, n_iterations=T, discard_s=args.discard,
                                            phase_screens=atm_file))
        f.attrs['summary'] = json.dumps(summary)
        f.attrs['description'] = __doc__
    logger.info(f"<<< {os.path.basename(res_path)}: OL rms {summary['ol_rms_px']:.4f} px | POL error {summary['pol_err_px']:.4f} | "
                + ' | '.join(f"{p}: OL input {summary[f'pred_err_ol_{p}_px']:.4f}, POL input {summary[f'pred_err_pol_{p}_px']:.4f} "
                             f"(quadrature {summary[f'quadrature_sum_{p}_px']:.4f})" for p in PREDICTORS)
                + f" | LP1-LP3 max diff {summary['lp1_lp3_max_diff_px']:.2e} px | {summary['s_per_it']:.3f} s/it")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    warnings.filterwarnings('ignore', message='Mean of empty slice')   # instants before the first prediction are NaN
    if args.test:
        args.n_iterations, args.draw, args.no_vibr_only = 100, 'draw1', True
        args.atm = None if args.mcao_profile1 else 'atm1'
        args.discard = 0.03
    log_helper = LoggingHelper(logging.INFO)
    logger = log_helper.logger
    base_dir = get_base_dir(args.base_dir)
    vibr = [False] if args.no_vibr_only else ([True] if args.vibr_only else [False, True])
    cases = {'mcaoP1': {'draw1': MCAO_PROFILE1}} if args.mcao_profile1 else atm_cases
    for atm_name, draws in cases.items():
        if args.atm is not None and atm_name != args.atm:
            continue
        for draw_name, kwargs in draws.items():
            if args.draw is not None and draw_name != args.draw:
                continue
            for use_vib in vibr:
                run_case(args, atm_name, draw_name, kwargs, use_vib, base_dir, logger)
    log_helper = None


if __name__ == '__main__':
    main()
