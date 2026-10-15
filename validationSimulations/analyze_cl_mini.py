"""
Summary of the reduced closed-loop runs of redArmSolarSCAO_05_CL_mini.py.

For every sensor and atmosphere: residual on-axis WFE of every configuration (rms over the iterations after the discarded transient),
change w.r.t. the best integrator gain and w.r.t. the best POL gain without predictor, and the closed-loop prediction diagnostics
(prediction, ZOH and POL errors w.r.t. the true open-loop slopes) of the runs with --diag.
The ceilings are the oracle runs: 'oracle' (perfect prediction and POL) and 'oracle_pol' (perfect prediction keeping the POL error).
"""
import os
import glob
import json
import argparse
import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Summary of the reduced closed-loop runs (cl_mini)')
    parser.add_argument('--results_dir', default=os.path.expanduser('~/simulations/results/cl_mini'))
    parser.add_argument('--csv', default=None, help='Optional CSV output (default <results_dir>/cl_mini_summary.csv)')
    args = parser.parse_args()

    runs = []
    for f in sorted(glob.glob(os.path.join(args.results_dir, 'mini_*.h5'))):
        with h5py.File(f, 'r') as h:
            c = json.loads(h.attrs['config'])
            s = json.loads(h.attrs['summary'])
        runs.append(dict(c, **s, file=os.path.basename(f)))
    if not runs:
        print(f'No runs in {args.results_dir}')
        return

    def label(r):
        if r['controller'] == 'integrator':
            return f"integrator g{r['gain']}"
        lab = f"POL {r['predictor']}"
        if r['predictor'] != 'none':
            lab += f" h{r['horizon']}"
        if r['predictor'] == 'stencil_rls':
            lab += f" lam{r['lam']}" + (' zero' if r['rls_init'] == 'zero' else '')
        return lab + f" g{r['gain']}"

    lines = ['sensor,atm,draw,wind_scale,config,wfe_nm,vs_best_integrator_pct,vs_best_pol_none_pct,diverged,cl_skill_pct,pred_err_px,zoh_err_px,pol_err_px,ol_rms_px']
    for r in runs:
        r.setdefault('wind_scale', 1.0)
    for key in sorted({(r['sensor'], r['atm'], r['draw'], r['wind_scale']) for r in runs}):
        sel = [r for r in runs if (r['sensor'], r['atm'], r['draw'], r['wind_scale']) == key]
        ok = [r for r in sel if not r['diverged']]
        integ = [r for r in ok if r['controller'] == 'integrator']
        poln = [r for r in ok if r['controller'] == 'pol' and r['predictor'] == 'none']
        best_i = min(integ, key=lambda r: r['wfe_mean_nm']) if integ else None
        best_p = min(poln, key=lambda r: r['wfe_mean_nm']) if poln else None
        print(f"\n=== {key[0]}x{key[0]} {key[1]} {key[2]} wind x{key[3]} ===  best integrator: {label(best_i) if best_i else '-'}"
              f"{' (%.1f nm)' % best_i['wfe_mean_nm'] if best_i else ''} | best POL without predictor: {label(best_p) if best_p else '-'}"
              f"{' (%.1f nm)' % best_p['wfe_mean_nm'] if best_p else ''}")
        print(f"{'configuration':<38} {'WFE [nm]':>9} {'vs integ':>9} {'vs POL':>8} | {'CL skill':>8} {'pred err':>9} {'ZOH err':>8} {'POL err':>8} {'OL rms':>7}")
        order = {'integrator': 0, 'none': 1, 'oracle': 2, 'oracle_pol': 3, 'stencil': 4, 'stencil_rls': 5, 'lstm': 6}
        for r in sorted(sel, key=lambda r: (order[r['predictor']] if r['controller'] == 'pol' else 0, r.get('horizon', 0), r['gain'])):
            w = r['wfe_mean_nm']
            vi = 100 * (w / best_i['wfe_mean_nm'] - 1) if best_i else np.nan
            vp = 100 * (w / best_p['wfe_mean_nm'] - 1) if best_p else np.nan
            d = r.get('cl_skill_pct')
            diag = (f"{d:8.1f} {r['pred_err']:9.4f} {r['zoh_err']:8.4f} {r['pol_err']:8.4f} {r['ol_rms']:7.3f}" if d is not None else '')
            print(f"{label(r):<38} {w:9.1f} {vi:+8.1f}% {vp:+7.1f}% | " + ('DIVERGED' if r['diverged'] else diag))
            lines.append(','.join(str(x) for x in [key[0], key[1], key[2], key[3], label(r), f'{w:.2f}', f'{vi:.2f}', f'{vp:.2f}', r['diverged'],
                                                    '' if d is None else f'{d:.2f}', r.get('pred_err', ''), r.get('zoh_err', ''),
                                                    r.get('pol_err', ''), r.get('ol_rms', '')]))
    out = args.csv or os.path.join(args.results_dir, 'cl_mini_summary.csv')
    with open(out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nCSV: {out}')


if __name__ == '__main__':
    main()
