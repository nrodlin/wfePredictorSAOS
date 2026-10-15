"""
Reduced closed-loop campaign to measure the ceilings of the prediction and to compare predictors before the full validation.

Configuration of 02/03 (EST red arm SCAO, 2 kHz, Correlating SH on the sun, ASM 36x36 or M7 50x50), but:
  - only the on-axis residual OPD rms is computed (NGS light path without science camera); no solar or PSF cameras
  - all the configurations of a sweep use the same atmosphere (paired comparison), with its own seeded phase screen
  - short runs (default 1000 iterations, first 200 discarded)

Controllers:
  integrator : leaky integrator on the residual slopes, c = g*(-R s_res) + decay*c
  pol        : POL law on a target (predicted) open-loop slope vector, c = g*(-R s_target - c) + decay*c, with s_target given by --predictor:
      none        s_pol(i-d)                                       (ZOH on the POL slopes)
      stencil     global stencil prediction of s_pol, horizon h
      stencil_rls adaptive stencil prediction of s_pol, horizon h
      lstm        current LSTM (trained for horizon 2)
      oracle      true open-loop slopes s_OL(i-d+h): perfect prediction and perfect POL. Measured with a second WFS on a copy of the
                  atmosphere advanced h-d updates (same phase screen file, per-layer seeded extrusion: identical future)
      oracle_pol  s_pol(i-d) + s_OL(i-d+h) - s_OL(i-d): perfect prediction of the increment, keeping the POL error

--inversion replaces the POL law by an inversion of the DM dynamics (same SISO state space for every mode, replicated by PseudoOpenLoop):
  the command c(i) first acts on the propagation i+2, y(i+2) = C A x(i) + C B c(i), so the target is the predicted modal shape for the
  atmosphere at i+2, i.e. horizon d+2 from the measurement s(i-d). This compensates the pure delay and the rise time of the DM.
      deadbeat : c(i) = (r(i+2) - C A x(i)) / (C B)
      reg      : min_c (y(i+2) - r(i+2))^2 + w3 (y(i+3) - r(i+3))^2 + rho_rel*(1 + w3 b3^2/b2^2)*b2^2 (c - c(i-1))^2, with c(i+1) = c(i),
                 targets at horizons d+2 and d+3
  --model_A_scale scales the A matrix of the DM model used by the POL replica and the inversion (robustness to a wrong DM model).

--diag adds an open-loop WFS on the current atmosphere and stores, after the predictor target time is reached, the rms of
  prediction error (s_target - s_OL(i-d+h)), ZOH error (s_pol(i-d) - s_OL(i-d+h)) and POL error (s_pol(i-d) - s_OL(i-d)).

Usage:
  python redArmSolarSCAO_05_CL_mini.py --check_oracle
  python redArmSolarSCAO_05_CL_mini.py --sweep ceilings [--sensors 36 50] [--atms atm2 atm4]
  python redArmSolarSCAO_05_CL_mini.py --controller pol --predictor stencil_rls --gain 1.0 --horizon 3 --atm atm2
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

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from SAOS.LoggingHelper import LoggingHelper
from SAOS.Source import Source
from SAOS.ExtendedSource import ExtendedSource
from SAOS.Telescope import Telescope
from SAOS.Atmosphere import Atmosphere
from SAOS.DeformableMirror import DeformableMirror
from SAOS.CorrelatingShackHartmann import CorrelatingShackHartmann
from SAOS.LightPath import LightPath
from SAOS.InteractionMatrixHandler import InteractionMatrixHandler
from SAOS.Controller import Controller

from wfePredictorSAOS.predictor.onlinePredictor import OnlineSlopePredictor
from wfePredictorSAOS.predictor.stencilPredictor import StencilPredictor
from wfePredictorSAOS.predictor.pseudoOpenLoop import PseudoOpenLoop

from redArmSolarSCAO_03_CL_POL import get_base_dir, get_asset_dirs
from atmosphereCases import atm_cases

PREDICTORS = ['none', 'stencil', 'stencil_rls', 'lstm', 'oracle', 'oracle_pol']


def sweep_configs(name, sensors, atms, draw):
    """Configurations of a sweep. Every configuration is run on every sensor and atmosphere."""
    base = []
    if name == 'ceilings':
        base += [dict(controller='integrator', predictor='none', gain=g) for g in (0.1, 0.15, 0.2, 0.3, 0.4, 0.5)]
        base += [dict(controller='pol', predictor='none', gain=g, diag=(g == 1.0)) for g in (0.3, 0.5, 0.7, 1.0)]
        base += [dict(controller='pol', predictor='oracle', gain=1.0, horizon=h) for h in (2, 3, 4)]
        base += [dict(controller='pol', predictor='oracle_pol', gain=g, horizon=3) for g in (0.7, 1.0)]
        base += [dict(controller='pol', predictor='stencil', gain=g, horizon=3, diag=True) for g in (0.7, 1.0)]
        base += [dict(controller='pol', predictor='stencil_rls', gain=g, horizon=3, diag=True) for g in (0.7, 1.0)]
        base += [dict(controller='pol', predictor='lstm', gain=1.0, horizon=2, diag=True)]
    elif name == 'wind':
        # Reduced set for the faster-wind cases (larger temporal error): best baselines, the realistic ceiling and the stencils
        base += [dict(controller='integrator', predictor='none', gain=g) for g in (0.1, 0.15, 0.2, 0.3)]
        base += [dict(controller='pol', predictor='none', gain=g, diag=(g == 1.0)) for g in (0.5, 1.0)]
        base += [dict(controller='pol', predictor='oracle_pol', gain=1.0, horizon=3)]
        base += [dict(controller='pol', predictor=p, gain=1.0, horizon=3, diag=True) for p in ('stencil', 'stencil_rls')]
    elif name == 'dm_inversion':
        # Inversion of the DM dynamics: ZOH target (dynamics only), realistic ceiling (oracle_pol) and stencils; robustness to the DM model
        for inv in ('deadbeat', 'reg'):
            base += [dict(controller='pol', predictor='none', inversion=inv, diag=True)]
            base += [dict(controller='pol', predictor='oracle_pol', inversion=inv)]
            base += [dict(controller='pol', predictor=p, inversion=inv, diag=True) for p in ('stencil', 'stencil_rls')]
        base += [dict(controller='pol', predictor='stencil', inversion='reg', model_A_scale=sc) for sc in (0.8, 1.2)]
    elif name == 'horizon':
        base += [dict(controller='pol', predictor=p, gain=1.0, horizon=h, diag=True) for p in ('stencil', 'stencil_rls') for h in (2, 4)]
        base += [dict(controller='pol', predictor='oracle_pol', gain=1.0, horizon=h) for h in (2, 4)]
    else:
        raise ValueError(f'Unknown sweep {name}')
    return [dict(c, sensor=s, atm=a, draw=draw) for s in sensors for a in atms for c in base]


def parse_args():
    parser = argparse.ArgumentParser(description="Reduced closed-loop runs: ceilings (oracles) and predictors, on-axis WFE only")
    parser.add_argument('--sweep', type=str, default=None, help="Run a predefined sweep (ceilings, horizon) instead of a single configuration")
    parser.add_argument('--list', action='store_true', help="List the configurations of the sweep and exit")
    parser.add_argument('--sensors', type=int, nargs='+', default=[36], help="Sensors of the sweep (default 36)")
    parser.add_argument('--atms', type=str, nargs='+', default=['atm2', 'atm4'], help="Atmospheres of the sweep (default atm2 atm4: LOS r0 8 and 15 cm)")
    parser.add_argument('--sensor', type=int, default=36, choices=[36, 50])
    parser.add_argument('--controller', type=str, default='pol', choices=['integrator', 'pol'])
    parser.add_argument('--predictor', type=str, default='none', choices=PREDICTORS)
    parser.add_argument('--gain', type=float, default=0.5)
    parser.add_argument('--decay', type=float, default=0.999)
    parser.add_argument('--delay', type=int, default=1, help="LightPath delay (total pure delay = delay + 1, plus the DM dynamics)")
    parser.add_argument('--horizon', type=int, default=3, help="Prediction horizon from the measurement time s(i-d) [samples]")
    parser.add_argument('--K', type=int, default=3, help="Stencil neighbourhood size")
    parser.add_argument('--p', type=int, default=8, help="Stencil past frames")
    parser.add_argument('--lam', type=float, default=0.995, help="Forgetting factor of the adaptive stencil")
    parser.add_argument('--rls_init', type=str, default='global', choices=['global', 'zero'])
    parser.add_argument('--inversion', type=str, default='none', choices=['none', 'deadbeat', 'reg'], help="Inversion of the DM dynamics instead of the POL law")
    parser.add_argument('--rho', type=float, default=0.1, help="Relative regularisation of the command increments (inversion 'reg')")
    parser.add_argument('--w3', type=float, default=0.5, help="Weight of the i+3 target (inversion 'reg')")
    parser.add_argument('--model_A_scale', type=float, default=1.0, help="Scale of the A matrix of the DM model used by POL and the inversion")
    parser.add_argument('--diag', action='store_true', help="Add an open-loop WFS to measure the prediction, ZOH and POL errors in closed loop")
    parser.add_argument('--atm', type=str, default='atm2')
    parser.add_argument('--draw', type=str, default='draw1')
    parser.add_argument('--wind_scale', type=float, default=1.0, help="Multiply all the wind speeds (own phase-screen file, same seed)")
    parser.add_argument('--n_iterations', type=int, default=1000)
    parser.add_argument('--discard', type=int, default=200, help="Iterations discarded in the summary (default 200 = 0.1 s)")
    parser.add_argument('--base_dir', type=str, default=None)
    parser.add_argument('--results_subdir', type=str, default='cl_mini')
    parser.add_argument('--skip_existing', action='store_true')
    parser.add_argument('--check_oracle', action='store_true', help="Check that the advanced atmosphere copy reproduces the future OPD and exit")
    parser.add_argument('--test', action='store_true', help="50 iterations, results in cl_mini_test")
    return parser.parse_args()


def run_name(c):
    name = f"mini_{c['sensor']}x{c['sensor']}_{c['atm']}_{c['draw']}" + (f"_wind{c['wind_scale']}" if c['wind_scale'] != 1 else "") + f"_{c['controller']}"
    if c['controller'] == 'pol':
        name += f"_{c['predictor']}"
        if c['predictor'] not in ('none',) and c.get('inversion', 'none') == 'none':
            name += f"_h{c['horizon']}"
        if c['predictor'] in ('stencil', 'stencil_rls'):
            name += f"_K{c['K']}p{c['p']}"
        if c['predictor'] == 'stencil_rls':
            name += f"_lam{c['lam']}_{c['rls_init']}"
    if c.get('inversion', 'none') != 'none':
        name += f"_inv{c['inversion']}" + (f"_rho{c['rho']}_w{c['w3']}" if c['inversion'] == 'reg' else '')
    else:
        name += f"_g{c['gain']}"
    if c.get('model_A_scale', 1.0) != 1.0:
        name += f"_Ascale{c['model_A_scale']}"
    name += f"_d{c['delay']}"
    return name + '.h5'


def load_atmosphere(tel, atm_name, draw_name, ps_dir, logger, wind_scale=1.0):
    """
    Validation atmosphere (atmosphereCases.py) with its own phase-screen file, generated with a fixed seed if it does not exist.
    wind_scale multiplies all the wind speeds (same seed, so the same screens moving faster). The winds are stored in the file.
    """
    kw = atm_cases[atm_name][draw_name]
    atm = Atmosphere(r0=kw['r0'], L0=kw['L0'], fractionalR0=kw['fractionalR0'], altitude=kw['altitude'], windDirection=kw['windDirection'],
                     windSpeed=[v * wind_scale for v in kw['windSpeed']], telescope=tel, zenith=kw['zenith'], logger=logger)
    path = os.path.join(ps_dir, f"ps_wfePredictor_mini_{atm_name}_{draw_name}" + (f"_wind{wind_scale}" if wind_scale != 1 else "") + ".h5")
    if not os.path.exists(path):
        seed = 100 * int(atm_name.replace('atm', '')) + int(draw_name.replace('draw', ''))
        atm.initializeAtmosphere(randomState=seed)
        atm.save(path)
        logger.info(f'Generated phase screens {path} (seed {seed})')
    if not atm.load(path):
        raise RuntimeError(f'Could not load {path}')
    return atm, path


def build_optics(sensor, tel, mirror_models_dir, logger):
    if sensor == 36:
        sun = ExtendedSource(optBand='R', coordinates=[0, 0], nSubDirs=3, fov=9.269, subDir_margin=4.0, patch_padding=5.0, logger=logger)
        ngs = Source(magnitude=5, optBand='R4', coordinates=[0, 0], logger=logger)
        dm = DeformableMirror(telescope=tel, nActs=37, altitude=0, typeDM='radial', logger=logger,
                              dynamicModel=os.path.join(mirror_models_dir, 'asm_discrete_model.h5'), validActThreshpercentage=0.7533)
        wfs_kw = dict(plate_scale=0.403, fieldOfView=9.269)
    else:
        sun = ExtendedSource(optBand='V', coordinates=[0, 0], nSubDirs=3, fov=9.975, subDir_margin=4.0, patch_padding=5.0, logger=logger)
        ngs = Source(magnitude=5, optBand='V0', coordinates=[0, 0], logger=logger)
        dm = DeformableMirror(telescope=tel, nActs=51, altitude=0, typeDM='cartesian', logger=logger,
                              dynamicModel=os.path.join(mirror_models_dir, 'm7_discrete_model.h5'), validActThreshpercentage=0.7533)
        wfs_kw = dict(plate_scale=0.475, fieldOfView=9.975)

    def make_wfs():
        return CorrelatingShackHartmann(telescope=tel, src=sun, lightRatio=0.9, nSubap=sensor, guardPx=2, fft_fieldOfView_oversampling=0.5,
                                        use_brightest=9, unit_in_rad=False, logger=logger, **wfs_kw)
    return sun, ngs, dm, make_wfs


def make_telescope(logger):
    return Telescope(diameter=4.149, resolution=200, centralObstruction=1.3 / 4.149, samplingTime=1 / 2000, fov=60.0, logger=logger)


def opd_rms_nm(opd, pupil):
    v = opd[pupil > 0]
    return float(np.std(v) * 1e9)


def check_oracle(args, logger):
    """The copy of the atmosphere advanced k updates must give, at iteration i, exactly the OPD of the original at i+k."""
    base_dir = get_base_dir(args.base_dir)
    ps_dir = os.path.join(base_dir, 'phase_screens')
    os.makedirs(ps_dir, exist_ok=True)
    tel = make_telescope(logger)
    ngs = Source(magnitude=5, optBand='R4', coordinates=[0, 0], logger=logger)
    sun = ExtendedSource(optBand='R', coordinates=[0, 0], nSubDirs=3, fov=9.269, subDir_margin=4.0, patch_padding=5.0, logger=logger)
    k, n = 2, 40
    atm, _ = load_atmosphere(tel, args.atm, args.draw, ps_dir, logger)
    atm_f, _ = load_atmosphere(tel, args.atm, args.draw, ps_dir, logger)
    for _ in range(k):
        atm_f.update()
    hist, hist_sun, fut, fut_sun, diffs, diffs_sun = [], [], [], [], [], []
    for i in range(n):
        atm.update()
        atm_f.update()
        hist.append(np.array(atm.getOPD(ngs)))
        hist_sun.append(np.array(atm.getOPD(sun)))
        fut.append(np.array(atm_f.getOPD(ngs)))
        fut_sun.append(np.array(atm_f.getOPD(sun)))
        if i >= k:
            # atm_f at iteration i-k must equal atm at iteration i
            diffs.append(np.max(np.abs(hist[i] - fut[i - k])))
            diffs_sun.append(np.max(np.abs(hist_sun[i] - fut_sun[i - k])))
    changes = [np.max(np.abs(hist[i] - hist[i - 1])) for i in range(1, n)]
    logger.info(f'Oracle check ({args.atm} {args.draw}, k={k}, {n} iterations): max |OPD_future(i-k) - OPD(i)| = {max(diffs):.3e} m (NGS), '
                f'{max(diffs_sun):.3e} m (sun sub-directions); typical OPD change per iteration {np.median(changes):.3e} m')
    return max(diffs) == 0 and max(diffs_sun) == 0


def run_one(c, args, logger, test=False):
    base_dir = get_base_dir(args.base_dir)
    mirror_models_dir, _ = get_asset_dirs(base_dir)
    ps_dir = os.path.join(base_dir, 'phase_screens')
    res_dir = os.path.join(base_dir, 'results', args.results_subdir + ('_test' if test else ''))
    os.makedirs(ps_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    res_path = os.path.join(res_dir, run_name(c))
    if args.skip_existing and os.path.exists(res_path):
        logger.info(f'Skipping {os.path.basename(res_path)}: already exists')
        return
    sensor, d, pred_name = c['sensor'], c['delay'], c['predictor']
    inversion = c.get('inversion', 'none')
    is_pol = c['controller'] == 'pol'
    # Prediction horizons from the measurement s(i-d): the POL law uses one (--horizon); the inversion needs the atmosphere at i+2 (and i+3)
    horizons = [c['horizon']] if inversion == 'none' else ([d + 2] if inversion == 'deadbeat' else [d + 2, d + 3])
    h = horizons[0]
    use_oracle = is_pol and pred_name in ('oracle', 'oracle_pol')
    if use_oracle and h < d:
        raise ValueError('The oracle needs horizon >= delay')
    if inversion != 'none' and (not is_pol or pred_name == 'lstm'):
        raise ValueError('The DM inversion is implemented for the POL controller with the none/stencil/oracle targets')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    t_start = time.time()

    tel = make_telescope(logger)
    atm, ps_path = load_atmosphere(tel, c['atm'], c['draw'], ps_dir, logger, c['wind_scale'])
    sun, ngs, dm, make_wfs = build_optics(sensor, tel, mirror_models_dir, logger)

    lp_wfs = LightPath(logger)
    lp_wfs.initialize_path(src=sun, atm=atm, tel=tel, dm=[dm], wfs=make_wfs(), delay=d)
    lp_sci = LightPath(logger)
    lp_sci.initialize_path(src=ngs, atm=atm, tel=tel, dm=[dm], wfs=None, delay=d)
    extra_paths = []
    if use_oracle:
        # Copy of the atmosphere advanced k_or = max(h)-d updates: at iteration i its open-loop slopes are s_OL(i+k_or)
        k_or = max(horizons) - d
        atm_f, _ = load_atmosphere(tel, c['atm'], c['draw'], ps_dir, logger, c['wind_scale'])
        for _ in range(k_or):
            atm_f.update()
        lp_or = LightPath(logger)
        lp_or.initialize_path(src=sun, atm=atm_f, tel=tel, dm=None, wfs=make_wfs(), delay=0)
        extra_paths.append(lp_or)
    if c['diag']:
        lp_ol = LightPath(logger)
        lp_ol.initialize_path(src=sun, atm=atm, tel=tel, dm=None, wfs=make_wfs(), delay=0)
        extra_paths.append(lp_ol)

    im_handler = InteractionMatrixHandler(logger)
    im_handler.initialize_im_class([lp_wfs, lp_sci])
    im_handler.load_modalBasis(os.path.join(base_dir, 'modal_basis', f'predictor_{sensor}x{sensor}_modalBasis.h5'))
    im_handler.load_IM(os.path.join(base_dir, 'interaction_matrix', f'predictor_{sensor}x{sensor}_IM.h5'))
    IM = im_handler.interaction_matrix_warehouse[0][0]['IM']
    controller = Controller(telescope=tel, interactionMatrix=im_handler, reconstructionMethod='tikhonov', controllerType='leaky', logger=logger,
                            rcond=0.025, beta=5e-4, gain=[c['gain']], decay=[c['decay']], ki=[0.0], nModes=[IM.shape[1]])
    n_modes = controller.reconstructor[0].shape[0]
    offset = controller.discarded_modes[0]
    R = controller.reconstructor[0].to(device)
    M = controller.modal_basis[0][:, offset:offset + n_modes].to(device)
    pol = PseudoOpenLoop(IM, dm, d, n_modes, device=device)
    if c.get('model_A_scale', 1.0) != 1.0:
        pol.A = pol.A * c['model_A_scale']                  # wrong DM model in the POL replica and in the inversion (the true DM is unchanged)
    n_slopes = lp_wfs.slopes_1D.shape[0]
    if inversion != 'none':
        if not pol.has_dynamics:
            raise ValueError('The DM inversion needs a DM with a dynamic model')
        CB = float(pol.C @ pol.B)                            # response at i+2 to c(i)
        CAB = float(pol.C @ pol.A @ pol.B)                   # response at i+3 to c(i)
        b2, b3, w3 = CB, CAB + CB, c['w3']                   # with c(i+1) = c(i) for the i+3 term
        rho_abs = c['rho'] * (b2 ** 2 + w3 * b3 ** 2)
        logger.info(f'DM inversion {inversion}: CB = {CB:.3f}, CAB = {CAB:.3f}, horizons {horizons}' + (f', rho {c["rho"]}, w3 {w3}' if inversion == 'reg' else ''))

    predictors = {}
    if pred_name in ('stencil', 'stencil_rls'):
        for hh in horizons:
            predictors[hh] = StencilPredictor(lp_wfs.wfs.valid_subapertures, K=c['K'], past_horizon=c['p'], horizon=hh,
                                              mode='global' if pred_name == 'stencil' else 'rls', lam=c['lam'], init=c['rls_init'], device=device)
    elif pred_name == 'lstm':
        if h != 2:
            raise ValueError('The current LSTM is trained for horizon 2')
        predictors[h] = OnlineSlopePredictor(n_slopes=n_slopes, past_horizon=24, hidden_size=16, n_axis=1)

    n_it = c['n_iterations']
    wfe = np.full(n_it, np.nan)
    res_rms = np.full(n_it, np.nan)
    diag = {k: np.full(n_it, np.nan) for k in ('pred_err', 'zoh_err', 'pol_err', 'ol_rms')} if c['diag'] else None
    cmd = torch.zeros((n_modes, 1), dtype=torch.float64, device=device)
    or_hist, target_hist, pol_hist, ol_hist = {}, {}, {}, {}
    logger.info(f">>> {run_name(c)} | atmosphere {ps_path}")
    t_loop = time.time()
    for i in range(n_it):
        atm.update()
        if use_oracle:
            atm_f.update()
        lp_wfs.propagate(True)
        lp_sci.propagate(True)
        for lp in extra_paths:
            lp.propagate(True)

        s_res = torch.as_tensor(lp_wfs.get_wavefront_error(), dtype=torch.float64, device=device).reshape(-1, 1)
        if c['controller'] == 'integrator':
            cmd = c['gain'] * (-(R @ s_res)) + c['decay'] * cmd
        else:
            s_pol = pol.reconstruct(s_res)
            # Predicted open-loop slopes for the atmosphere at i-d+hh, for every horizon (ZOH = s_pol until the predictors are ready)
            targets = {hh: s_pol for hh in horizons}
            if predictors:
                s_np = s_pol.squeeze(1).cpu().numpy()
                for hh, pr in predictors.items():
                    pr.push(s_np)
                    if pr.ready():
                        targets[hh] = torch.as_tensor(pr.predict(), dtype=torch.float64, device=device).reshape(-1, 1)
            elif use_oracle:
                or_hist[i + k_or] = torch.as_tensor(lp_or.slopes_1D.copy(), dtype=torch.float64, device=device).reshape(-1, 1)
                for hh in horizons:
                    if (i - d + hh) not in or_hist:
                        continue
                    if pred_name == 'oracle':
                        targets[hh] = or_hist[i - d + hh]
                    elif (i - d) in or_hist:
                        targets[hh] = s_pol + or_hist[i - d + hh] - or_hist[i - d]
                or_hist.pop(i - d - 1, None)
            target = targets[h]
            if inversion == 'none':
                cmd = c['gain'] * (-(R @ target) - cmd) + c['decay'] * cmd
            else:
                xA = pol.state @ pol.A.T                                    # A x(i), x(i) = DM state before c(i)
                a2 = xA @ pol.C.T                                          # free response at i+2
                r2 = -(R @ targets[horizons[0]])
                if inversion == 'deadbeat':
                    cmd = (r2 - a2) / b2
                else:
                    a3 = (xA @ pol.A.T) @ pol.C.T                          # free response at i+3
                    r3 = -(R @ targets[horizons[1]])
                    cmd = (b2 * (r2 - a2) + w3 * b3 * (r3 - a3) + rho_abs * cmd) / (b2 ** 2 + w3 * b3 ** 2 + rho_abs)
            if diag is not None:
                # Targets refer to the atmosphere at i-d+h, the POL slopes to i-d. The open-loop truth of iteration j is s_OL(j)
                target_hist[i - d + h] = target.squeeze(1).cpu().numpy()
                pol_hist[i - d] = s_pol.squeeze(1).cpu().numpy()
        dm.updateDMShape(M @ cmd)
        pol.update(cmd)

        wfe[i] = opd_rms_nm(np.asarray(lp_sci.sci_opd), tel.pupil)
        res_rms[i] = float(s_res.std())
        if diag is not None:
            s_ol = lp_ol.slopes_1D.copy()
            diag['ol_rms'][i] = float(np.sqrt(np.mean(s_ol ** 2)))
            if i in target_hist:
                diag['pred_err'][i] = float(np.sqrt(np.mean((target_hist.pop(i) - s_ol) ** 2)))
            if (i - h) in pol_hist:                              # ZOH: POL slopes of the atmosphere at i-h used for the target time i
                diag['zoh_err'][i] = float(np.sqrt(np.mean((pol_hist[i - h] - s_ol) ** 2)))
            ol_hist[i] = s_ol
            if (i - d) in pol_hist and (i - d) in ol_hist:      # POL slopes of the atmosphere at i-d arrive at iteration i
                diag['pol_err'][i] = float(np.sqrt(np.mean((pol_hist[i - d] - ol_hist[i - d]) ** 2)))
            pol_hist.pop(i - h - 1, None)
            ol_hist.pop(i - d - 1, None)
        if not np.isfinite(wfe[i]) or wfe[i] > 1e5:
            logger.warning(f'  loop diverged at iteration {i} (WFE {wfe[i]:.3e} nm)')
            break
        if (i + 1) % 250 == 0 or i == 0:
            logger.info(f'  [it {i + 1:5d}/{n_it}] WFE {wfe[i]:7.1f} nm | residual slopes rms {res_rms[i]:.4f} px | {(time.time() - t_loop) / (i + 1):.3f} s/it')

    k0 = c['discard']
    summary = dict(wfe_mean_nm=float(np.sqrt(np.nanmean(wfe[k0:] ** 2))), diverged=bool(np.isnan(wfe).any() or (wfe[np.isfinite(wfe)] > 1e5).any()),
                   s_per_it=(time.time() - t_loop) / max(1, np.isfinite(wfe).sum()), wall_s=time.time() - t_start)
    if diag is not None:
        for k in ('pred_err', 'zoh_err', 'pol_err', 'ol_rms'):
            summary[k] = float(np.sqrt(np.nanmean(diag[k][k0:] ** 2)))
        summary['cl_skill_pct'] = float(100 * (1 - summary['pred_err'] ** 2 / summary['zoh_err'] ** 2))
    with h5py.File(res_path, 'w') as f:
        f.create_dataset('wfe_nm', data=wfe)
        f.create_dataset('res_slopes_rms', data=res_rms)
        if diag is not None:
            for k, v in diag.items():
                f.create_dataset(f'diag/{k}', data=v)
        f.attrs['config'] = json.dumps(c)
        f.attrs['summary'] = json.dumps(summary)
        f.attrs['phase_screens'] = ps_path
    msg = f"<<< {run_name(c)}: WFE {summary['wfe_mean_nm']:.1f} nm (it >= {k0}){' DIVERGED' if summary['diverged'] else ''}, {summary['s_per_it']:.3f} s/it"
    if diag is not None:
        msg += (f" | pred err {summary['pred_err']:.4f}, ZOH err {summary['zoh_err']:.4f}, POL err {summary['pol_err']:.4f}, "
                f"OL rms {summary['ol_rms']:.4f} px, CL skill {summary['cl_skill_pct']:.1f} %")
    logger.info(msg)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    log_helper = LoggingHelper(logging.INFO)          # keep the reference: the queue listener stops when it is deleted
    logger = log_helper.logger
    if args.check_oracle:
        ok = check_oracle(args, logger)
        logger.info('Oracle check ' + ('PASSED' if ok else 'FAILED'))
        log_helper = None
        return
    common = dict(wind_scale=args.wind_scale, inversion=args.inversion, rho=args.rho, w3=args.w3, model_A_scale=args.model_A_scale, decay=args.decay, delay=args.delay, K=args.K, p=args.p, lam=args.lam, rls_init=args.rls_init,
                  n_iterations=50 if args.test else args.n_iterations, discard=10 if args.test else args.discard)
    if args.sweep:
        configs = []
        for cfg in sweep_configs(args.sweep, args.sensors, args.atms, args.draw):
            full = dict(common, horizon=args.horizon, diag=False, gain=1.0)   # gain is not used by the DM inversion
            full.update(cfg)
            configs.append(full)
    else:
        configs = [dict(common, sensor=args.sensor, controller=args.controller, predictor=args.predictor, gain=args.gain, horizon=args.horizon,
                        diag=args.diag, atm=args.atm, draw=args.draw)]
    if args.list:
        for i, cfg in enumerate(configs):
            print(f'{i + 1:3d} {run_name(cfg)}' + (' [diag]' if cfg['diag'] else ''))
        return
    logger.info(f'{len(configs)} configuration(s)')
    for i, cfg in enumerate(configs):
        logger.info(f'=== [{i + 1}/{len(configs)}] ===')
        run_one(cfg, args, logger, test=args.test)
    log_helper = None


if __name__ == '__main__':
    main()
