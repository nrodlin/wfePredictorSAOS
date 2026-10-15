"""
Paired closed-loop comparison of control strategies on the same atmosphere (EST red arm SCAO, 2 kHz, Correlating SH on the sun).

Cases (all with the same phase screens, DM, IM and delay; noiseless WFS, no vibrations):
  integrator      leaky integrator on the residual slopes, c = g_int*(-R s_res) + decay*c (no POL)
  pol:<pred>      POL law c = g*(-R s_target - c) + decay*c, with s_target the prediction of the POL slopes by <pred>
                  (common.make_predictor names: zoh, linear, lstm_v1, lstm_v2[:tag], incAR:<file>, stencil_rls). All the POL cases
                  use the same gain g. While a predictor fills its buffer the target is s_pol (ZOH), so until then the POL cases are identical.

Metric: on-axis residual OPD rms of an NGS light path (no science cameras). Results: one h5 per case in
<base_dir>/results/closedLoopComparison/<run tag>/, plus summary table (csv) and figure (WFE vs iteration).

Usage:
  python closedLoopComparison.py                                   # 36x36, atm1 draw1, 500 it, POL g 0.4, integrator g 0.25
  python closedLoopComparison.py --cases pol:zoh pol:linear pol:lstm_v2 --horizon 2
  python closedLoopComparison.py --analyze_only
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from SAOS.LoggingHelper import LoggingHelper
from SAOS.LightPath import LightPath

from common import get_base_dir, get_asset_dirs, make_telescope, build_optics, load_atmosphere, PolControl, make_predictor, opd_rms_nm, MCAO_PROFILE1
from atmosphereCases import atm_cases

DEFAULT_CASES = ['integrator', 'pol:zoh', 'pol:linear', 'pol:lstm_v1']
COLORS = {'integrator': '#7f7f7f', 'pol:zoh': '#2a78d6', 'pol:lstm_v1': '#d62728', 'pol:linear': '#1baf7a', 'pol:lstm_v2': '#9467bd'}


def parse_args():
    parser = argparse.ArgumentParser(description="Paired closed-loop comparison: integrator and POL with several predictors")
    parser.add_argument('--cases', nargs='+', default=DEFAULT_CASES)
    parser.add_argument('--sensor', type=int, default=36, choices=[36, 50])
    parser.add_argument('--atm', type=str, default='atm1', help="Validation atmosphere of atmosphereCases.py (ignored with --mcao_profile1)")
    parser.add_argument('--draw', type=str, default='draw1')
    parser.add_argument('--mcao_profile1', action='store_true', help="Profile 1 of mcaobenchcodes instead of a validation atmosphere")
    parser.add_argument('--n_iterations', type=int, default=500)
    parser.add_argument('--discard', type=int, default=100, help="Iterations discarded in the summary (default 100 = 0.05 s)")
    parser.add_argument('--gain', type=float, default=0.4, help="Gain of the POL law, the same for every POL case (default 0.4)")
    parser.add_argument('--int_gain', type=float, default=0.25, help="Gain of the integrator (default 0.25)")
    parser.add_argument('--decay', type=float, default=0.999)
    parser.add_argument('--delay', type=int, default=1, help="LightPath delay (total pure delay = delay + 1)")
    parser.add_argument('--horizon', type=int, default=2, help="Prediction horizon of the predictors")
    parser.add_argument('--linear_past', type=int, default=4)
    parser.add_argument('--beta', type=float, default=5e-4)
    parser.add_argument('--base_dir', type=str, default=None)
    parser.add_argument('--skip_existing', action='store_true')
    parser.add_argument('--analyze_only', action='store_true')
    parser.add_argument('--test', action='store_true', help="30 iterations, results in a _test folder")
    return parser.parse_args()


def atmosphere_choice(args):
    """(name, parameters, phase-screen file name, seed). The validation cases keep the files and seeds of the first comparison."""
    if args.mcao_profile1:
        return 'mcaoP1', MCAO_PROFILE1, 'ps_polpred_mcaoProfile1_draw1_seed2026.h5', 2026
    seed = 100 * int(args.atm.replace('atm', '')) + int(args.draw.replace('draw', ''))
    return f'{args.atm}_{args.draw}', atm_cases[args.atm][args.draw], f'ps_wfePredictor_mini_{args.atm}_{args.draw}.h5', seed


def run_tag(args):
    return (f"{args.sensor}x{args.sensor}_{atmosphere_choice(args)[0]}_noVibr_noNoise_{args.n_iterations}it_d{args.delay}_h{args.horizon}"
            f"_polg{args.gain}_intg{args.int_gain}" + ('_test' if args.test else ''))


def case_file(case):
    return 'res_' + case.replace(':', '_') + '.h5'


def run_case(case, args, res_dir, logger):
    res_path = os.path.join(res_dir, case_file(case))
    if args.skip_existing and os.path.exists(res_path):
        logger.info(f'Skipping {case}: {res_path} exists')
        return
    if case != 'integrator' and not case.startswith('pol:'):
        raise ValueError(f'Unknown case {case}')
    base_dir = get_base_dir(args.base_dir)
    mirror_models_dir, _ = get_asset_dirs(base_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    s, d = args.sensor, args.delay
    gain = args.int_gain if case == 'integrator' else args.gain
    t_start = time.time()

    tel = make_telescope(logger)
    _, atm_kw, ps_name, seed = atmosphere_choice(args)
    ps_path = os.path.join(base_dir, 'phase_screens', ps_name)
    atm = load_atmosphere(tel, atm_kw, ps_path, seed, logger)
    sun, ngs, dm, make_wfs = build_optics(s, tel, mirror_models_dir, logger)
    wfs = make_wfs()
    lp_wfs = LightPath(logger)
    lp_wfs.initialize_path(src=sun, atm=atm, tel=tel, dm=[dm], wfs=wfs, delay=d)
    lp_sci = LightPath(logger)
    lp_sci.initialize_path(src=ngs, atm=atm, tel=tel, dm=[dm], wfs=None, delay=d)
    ctrl = PolControl(base_dir, s, [lp_wfs, lp_sci], dm, d, tel, logger, beta=args.beta, decay=args.decay, device=device)
    n_slopes = lp_wfs.slopes_1D.shape[0]
    predictor = None
    if case.startswith('pol:'):
        predictor = make_predictor(case[4:], n_slopes, args.horizon, valid_subapertures=wfs.valid_subapertures, linear_past=args.linear_past, device=device)

    n_it = args.n_iterations
    wfe = np.full(n_it, np.nan)
    res_rms = np.full(n_it, np.nan)
    pred_active = np.zeros(n_it, dtype=bool)
    logger.info(f'>>> {case} | gain {gain}, decay {args.decay}, delay {d}, horizon {args.horizon} | {s}x{s} | phase screens {ps_path}')
    t_loop = time.time()
    for i in range(n_it):
        atm.update()
        lp_wfs.propagate(True)
        lp_sci.propagate(True)
        s_res = ctrl.tensor(lp_wfs.get_wavefront_error())
        if case == 'integrator':
            ctrl.step_integrator(s_res, gain)
        else:
            s_pol = ctrl.pol_slopes(s_res)
            target = s_pol
            if predictor is not None:
                predictor.push(s_pol.squeeze(1).cpu().numpy())
                if predictor.ready():
                    target = ctrl.tensor(predictor.predict())
                    pred_active[i] = True
            ctrl.step_pol(target, gain)

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
    for p in sorted(glob.glob(os.path.join(res_dir, 'res_*.h5'))):
        with h5py.File(p, 'r') as f:
            sm = json.loads(f.attrs['summary'])
            res[sm['case']] = dict(wfe=f['wfe_nm'][:], summary=sm)
    if not res:
        print(f'No results in {res_dir}')
        return
    order = [c for c in DEFAULT_CASES if c in res] + sorted(c for c in res if c not in DEFAULT_CASES)
    ref = res.get('pol:zoh', {}).get('summary', {}).get('wfe_nm')
    lines = [f"{'case':28s} {'gain':>5s} {'WFE [nm]':>9s} {'vs POL ZOH':>11s} {'res. slopes [px]':>17s}"]
    rows = ['case,gain,wfe_nm,rel_pol_zoh_pct,res_slopes_rms_px,diverged']
    for case in order:
        sm = res[case]['summary']
        rel = 100 * (sm['wfe_nm'] / ref - 1) if ref else float('nan')
        lines.append(f"{case:28s} {sm['gain']:5.2f} {sm['wfe_nm']:9.1f} {rel:+10.1f}% {sm['res_slopes_rms_px']:17.4f}" + (' DIVERGED' if sm['diverged'] else ''))
        rows.append(f"{case},{sm['gain']},{sm['wfe_nm']:.4f},{rel:.3f},{sm['res_slopes_rms_px']:.6f},{sm['diverged']}")
    print(f'Iterations >= {args.discard} | {res_dir}')
    print('\n'.join(lines))
    with open(os.path.join(res_dir, 'summary.csv'), 'w') as f:
        f.write('\n'.join(rows) + '\n')

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for case in order:
        r = res[case]
        ax.plot(r['wfe'], color=COLORS.get(case), lw=1.2, label=f"{case} (g {r['summary']['gain']}): {r['summary']['wfe_nm']:.1f} nm")
    ax.axvspan(0, args.discard, color='0.92', zorder=0)
    ax.set_xlabel('Iteration (2 kHz)')
    ax.set_ylabel('On-axis residual WFE [nm]')
    ax.set_title(f'{args.sensor}x{args.sensor}, {atmosphere_choice(args)[0]}, no vibrations, no noise. Shaded: discarded')
    ax.legend(frameon=False, fontsize=8)
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
    res_dir = os.path.join(base_dir, 'results', 'closedLoopComparison', run_tag(args))
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
