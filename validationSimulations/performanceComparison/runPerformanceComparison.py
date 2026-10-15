"""
Paired closed-loop comparison of the control strategies on the same atmosphere (EST red arm SCAO, 2 kHz, Correlating SH on the sun).

Cases (all with the same phase screens, DM, IM and delay; the WFS has no noise and there are no vibrations):
  integrator  : leaky integrator on the residual slopes, c = g_int*(-R s_res) + decay*c (no POL)
  pol_none    : POL law with the POL slopes as target (ZOH), c = g*(-R s_pol - c) + decay*c
  pol_lstm    : POL law with the LSTM prediction of the POL slopes (horizon 2) as target
  pol_linear  : POL law with the linear extrapolation of the POL slopes (past 4, horizon 2, as in 03) as target
  pol_lstm_dc : as pol_lstm, with the LSTM output divided by its gain at zero frequency (--lstm_dc_gain, 0.975 measured offline)
  pol_incAR_* : POL law with the increment AR predictor of the POL slopes (predictor/incrementARPredictor.py), weights file
                predictor/<name>.npz, e.g. pol_incAR_p16 or pol_incAR_p16_lp50_hf0.1 (horizon --horizon)
The three POL cases use the same gain g. While the predictor buffer fills, the POL target is s_pol (ZOH), so until then the
three POL cases are identical.

Configuration of 05 (telescope, optics, IM, POL replica of the DM dynamics). Metric: on-axis residual OPD rms of an NGS light path
(no science cameras). Results: one h5 per case in <base_dir>/results/performanceComparison/<run tag>/, plus summary table (csv)
and figure (WFE vs iteration).

Usage:
  python runPerformanceComparison.py                       # 36x36, atm1 draw1, 500 iterations, POL gain 0.4, integrator gain 0.25
  python runPerformanceComparison.py --cases pol_none pol_lstm --gain 0.4
  python runPerformanceComparison.py --analyze_only        # table and figure from the existing results
"""
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import sys
import time
import json
import glob
import argparse
import logging
import h5py
import numpy as np
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
validation_dir = os.path.dirname(script_dir)
if validation_dir not in sys.path:
    sys.path.insert(0, validation_dir)

from SAOS.LoggingHelper import LoggingHelper
from SAOS.LightPath import LightPath
from SAOS.InteractionMatrixHandler import InteractionMatrixHandler
from SAOS.Controller import Controller

from wfePredictorSAOS.predictor.onlinePredictor import OnlineSlopePredictor
from wfePredictorSAOS.predictor.onlineLinearPredictor import OnlineLinearSlopePredictor
from wfePredictorSAOS.predictor.pseudoOpenLoop import PseudoOpenLoop
from wfePredictorSAOS.predictor.incrementARPredictor import IncrementARPredictor

from redArmSolarSCAO_03_CL_POL import get_base_dir, get_asset_dirs
from redArmSolarSCAO_05_CL_mini import load_atmosphere, build_optics, make_telescope, opd_rms_nm

CASES = ('integrator', 'pol_none', 'pol_lstm', 'pol_linear')
COLORS = {'integrator': '#7f7f7f', 'pol_none': '#2a78d6', 'pol_lstm': '#d62728', 'pol_linear': '#1baf7a', 'pol_lstm_dc': '#ff9896'}
PREDICTOR_DIR = os.path.join(os.path.dirname(validation_dir), 'wfePredictorSAOS', 'predictor')


def valid_case(case):
    if case in CASES or case == 'pol_lstm_dc':
        return case
    if case.startswith('pol_incAR') and os.path.exists(os.path.join(PREDICTOR_DIR, case[4:] + '.npz')):
        return case
    raise argparse.ArgumentTypeError(f'Unknown case {case} (or missing weights {case[4:]}.npz in {PREDICTOR_DIR})')


def parse_args():
    parser = argparse.ArgumentParser(description="Paired closed-loop comparison: integrator, POL, POL+LSTM, POL+linear")
    parser.add_argument('--cases', type=valid_case, nargs='+', default=list(CASES), help=f"Cases (default {' '.join(CASES)}; also pol_lstm_dc, pol_incAR_*)")
    parser.add_argument('--lstm_dc_gain', type=float, default=0.975, help="Gain at zero frequency of the LSTM, corrected in pol_lstm_dc (default 0.975)")
    parser.add_argument('--sensor', type=int, default=36, choices=[36, 50])
    parser.add_argument('--atm', type=str, default='atm1')
    parser.add_argument('--draw', type=str, default='draw1')
    parser.add_argument('--n_iterations', type=int, default=500)
    parser.add_argument('--discard', type=int, default=100, help="Iterations discarded in the summary (default 100 = 0.05 s)")
    parser.add_argument('--gain', type=float, default=0.4, help="Gain of the POL law, the same for the three POL cases (default 0.4)")
    parser.add_argument('--int_gain', type=float, default=0.25, help="Gain of the integrator (default 0.25, best 36x36 integrator of fourthAttempt)")
    parser.add_argument('--decay', type=float, default=0.999)
    parser.add_argument('--delay', type=int, default=1, help="LightPath delay (total pure delay = delay + 1)")
    parser.add_argument('--horizon', type=int, default=2, help="Horizon of the linear extrapolation (the LSTM is trained for 2)")
    parser.add_argument('--linear_past', type=int, default=4, help="Past samples of the linear extrapolation (default 4, as in 03)")
    parser.add_argument('--beta', type=float, default=5e-4, help="Tikhonov regularisation")
    parser.add_argument('--base_dir', type=str, default=None)
    parser.add_argument('--skip_existing', action='store_true')
    parser.add_argument('--analyze_only', action='store_true')
    parser.add_argument('--test', action='store_true', help="30 iterations, results in a _test folder")
    return parser.parse_args()


def run_tag(args):
    return (f"{args.sensor}x{args.sensor}_{args.atm}_{args.draw}_noVibr_noNoise_{args.n_iterations}it_d{args.delay}"
            f"_polg{args.gain}_intg{args.int_gain}" + ('_test' if args.test else ''))


def case_gain(case, args):
    return args.int_gain if case == 'integrator' else args.gain


def run_case(case, args, res_dir, logger):
    res_path = os.path.join(res_dir, f'res_{case}.h5')
    if args.skip_existing and os.path.exists(res_path):
        logger.info(f'Skipping {case}: {res_path} exists')
        return
    base_dir = get_base_dir(args.base_dir)
    mirror_models_dir, _ = get_asset_dirs(base_dir)
    ps_dir = os.path.join(base_dir, 'phase_screens')
    os.makedirs(ps_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    s, d, gain = args.sensor, args.delay, case_gain(case, args)
    t_start = time.time()

    # Same seeded phase-screen file for every case (shared with the 05 mini-CL runs of this atmosphere)
    tel = make_telescope(logger)
    atm, ps_path = load_atmosphere(tel, args.atm, args.draw, ps_dir, logger)
    sun, ngs, dm, make_wfs = build_optics(s, tel, mirror_models_dir, logger)
    wfs = make_wfs()                                  # CorrelatingShackHartmann default noiseFlag=False: noiseless detector

    lp_wfs = LightPath(logger)
    lp_wfs.initialize_path(src=sun, atm=atm, tel=tel, dm=[dm], wfs=wfs, delay=d)
    lp_sci = LightPath(logger)
    lp_sci.initialize_path(src=ngs, atm=atm, tel=tel, dm=[dm], wfs=None, delay=d)

    im_handler = InteractionMatrixHandler(logger)
    im_handler.initialize_im_class([lp_wfs, lp_sci])
    im_handler.load_modalBasis(os.path.join(base_dir, 'modal_basis', f'predictor_{s}x{s}_modalBasis.h5'))
    im_handler.load_IM(os.path.join(base_dir, 'interaction_matrix', f'predictor_{s}x{s}_IM.h5'))
    IM = im_handler.interaction_matrix_warehouse[0][0]['IM']
    controller = Controller(telescope=tel, interactionMatrix=im_handler, reconstructionMethod='tikhonov', controllerType='leaky', logger=logger,
                            rcond=0.025, beta=args.beta, gain=[gain], decay=[args.decay], ki=[0.0], nModes=[IM.shape[1]])
    n_modes = controller.reconstructor[0].shape[0]
    offset = controller.discarded_modes[0]
    R = controller.reconstructor[0].to(device)
    M = controller.modal_basis[0][:, offset:offset + n_modes].to(device)
    pol = PseudoOpenLoop(IM, dm, d, n_modes, device=device)
    n_slopes = lp_wfs.slopes_1D.shape[0]

    if case in ('pol_lstm', 'pol_lstm_dc'):
        predictor = OnlineSlopePredictor(n_slopes=n_slopes, past_horizon=24, hidden_size=16, n_axis=1)
    elif case.startswith('pol_incAR'):
        predictor = IncrementARPredictor(n_slopes=n_slopes, weights_file=os.path.join(PREDICTOR_DIR, case[4:] + '.npz'), horizon=args.horizon)
    elif case == 'pol_linear':
        predictor = OnlineLinearSlopePredictor(n_slopes=n_slopes, past_horizon=args.linear_past, steps_ahead=args.horizon)
    else:
        predictor = None

    n_it = args.n_iterations
    wfe = np.full(n_it, np.nan)
    res_rms = np.full(n_it, np.nan)
    pred_active = np.zeros(n_it, dtype=bool)
    cmd = torch.zeros((n_modes, 1), dtype=torch.float64, device=device)
    logger.info(f'>>> {case} | gain {gain}, decay {args.decay}, delay {d} | {s}x{s} | phase screens {ps_path}')
    t_loop = time.time()
    for i in range(n_it):
        atm.update()
        lp_wfs.propagate(True)
        lp_sci.propagate(True)

        s_res = torch.as_tensor(lp_wfs.get_wavefront_error(), dtype=torch.float64, device=device).reshape(-1, 1)
        if case == 'integrator':
            cmd = gain * (-(R @ s_res)) + args.decay * cmd
        else:
            s_pol = pol.reconstruct(s_res)
            target = s_pol
            if predictor is not None:
                predictor.push(s_pol.squeeze(1).cpu().numpy())
                if predictor.ready():
                    target = torch.as_tensor(predictor.predict(), dtype=torch.float64, device=device).reshape(-1, 1)
                    if case == 'pol_lstm_dc':
                        target = target / args.lstm_dc_gain
                    pred_active[i] = True
            cmd = gain * (-(R @ target) - cmd) + args.decay * cmd
        dm.updateDMShape(M @ cmd)
        pol.update(cmd)

        wfe[i] = opd_rms_nm(np.asarray(lp_sci.sci_opd), tel.pupil)
        res_rms[i] = float(s_res.std())
        if not np.isfinite(wfe[i]) or wfe[i] > 1e5:
            logger.warning(f'  loop diverged at iteration {i} (WFE {wfe[i]:.3e} nm)')
            break
        if (i + 1) % 100 == 0 or i == 0:
            logger.info(f'  [it {i + 1:4d}/{n_it}] WFE {wfe[i]:7.1f} nm | residual slopes rms {res_rms[i]:.4f} px | {(time.time() - t_loop) / (i + 1):.3f} s/it')

    k0 = min(args.discard, n_it - 1)
    summary = dict(case=case, gain=gain, wfe_nm=float(np.sqrt(np.nanmean(wfe[k0:] ** 2))), res_slopes_rms_px=float(np.sqrt(np.nanmean(res_rms[k0:] ** 2))),
                   diverged=bool(not np.isfinite(wfe).all() or np.nanmax(wfe) > 1e5),
                   first_predicted_it=int(np.argmax(pred_active)) if pred_active.any() else None,
                   s_per_it=(time.time() - t_loop) / max(1, np.isfinite(wfe).sum()), wall_s=time.time() - t_start)
    with h5py.File(res_path, 'w') as f:
        f.create_dataset('wfe_nm', data=wfe)
        f.create_dataset('res_slopes_rms', data=res_rms)
        f.create_dataset('predictor_active', data=pred_active)
        f.attrs['config'] = json.dumps(dict(vars(args), case=case, case_gain=gain))
        f.attrs['summary'] = json.dumps(summary)
        f.attrs['phase_screens'] = ps_path
    logger.info(f"<<< {case}: WFE {summary['wfe_nm']:.1f} nm (it >= {k0}){' DIVERGED' if summary['diverged'] else ''}, {summary['s_per_it']:.3f} s/it")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def analyze(args, res_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    res = {}
    found = sorted(os.path.basename(p)[4:-3] for p in glob.glob(os.path.join(res_dir, 'res_*.h5')))
    for case in [c for c in CASES if c in found] + [c for c in found if c not in CASES]:
        with h5py.File(os.path.join(res_dir, f'res_{case}.h5'), 'r') as f:
            res[case] = dict(wfe=f['wfe_nm'][:], summary=json.loads(f.attrs['summary']))
    if not res:
        print(f'No results in {res_dir}')
        return
    ref = res.get('pol_none', {}).get('summary', {}).get('wfe_nm')
    lines = [f"{'case':26s} {'gain':>5s} {'WFE [nm]':>9s} {'vs POL none':>12s} {'res. slopes [px]':>17s}"]
    rows = ['case,gain,wfe_nm,rel_pol_none_pct,res_slopes_rms_px,diverged']
    for case, r in res.items():
        sm = r['summary']
        rel = 100 * (sm['wfe_nm'] / ref - 1) if ref else float('nan')
        lines.append(f"{case:26s} {sm['gain']:5.2f} {sm['wfe_nm']:9.1f} {rel:+11.1f}% {sm['res_slopes_rms_px']:17.4f}" + (' DIVERGED' if sm['diverged'] else ''))
        rows.append(f"{case},{sm['gain']},{sm['wfe_nm']:.4f},{rel:.3f},{sm['res_slopes_rms_px']:.6f},{sm['diverged']}")
    print(f'Iterations >= {args.discard} | {res_dir}')
    print('\n'.join(lines))
    with open(os.path.join(res_dir, 'summary.csv'), 'w') as f:
        f.write('\n'.join(rows) + '\n')

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for case, r in res.items():
        ax.plot(r['wfe'], color=COLORS.get(case), lw=1.2, label=f"{case} (g {r['summary']['gain']}): {r['summary']['wfe_nm']:.1f} nm")
    ax.axvspan(0, args.discard, color='0.92', zorder=0)
    ax.set_xlabel('Iteration (2 kHz)')
    ax.set_ylabel('On-axis residual WFE [nm]')
    ax.set_title(f'{args.sensor}x{args.sensor}, {args.atm} {args.draw}, no vibrations, no noise. Shaded: discarded')
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, color='0.9')
    ax.spines[['right', 'top']].set_visible(False)
    finite = np.concatenate([r['wfe'][args.discard:][np.isfinite(r['wfe'][args.discard:])] for r in res.values()])
    if finite.size:
        ax.set_ylim(0, 1.3 * np.percentile(finite, 99.5))
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(res_dir, f'wfe_comparison.{ext}'), dpi=200)
    print(f'Saved summary.csv and wfe_comparison.png/pdf in {res_dir}')


def main():
    args = parse_args()
    if args.test:
        args.n_iterations, args.discard = 30, 5
    base_dir = get_base_dir(args.base_dir)
    res_dir = os.path.join(base_dir, 'results', 'performanceComparison', run_tag(args))
    os.makedirs(res_dir, exist_ok=True)
    if not args.analyze_only:
        log_helper = LoggingHelper(logging.INFO)      # keep the reference: the queue listener stops when it is deleted
        logger = log_helper.logger
        for k, case in enumerate(args.cases):
            logger.info(f'=== [{k + 1}/{len(args.cases)}] {case} ===')
            run_case(case, args, res_dir, logger)
        log_helper = None
    analyze(args, res_dir)


if __name__ == '__main__':
    main()
