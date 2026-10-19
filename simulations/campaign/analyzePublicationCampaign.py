#!/usr/bin/env python3
"""
Statistics of the publication campaign (runPublicationCampaign.py) for the paper.

Closed loop (closedLoopComparison, 1000 iterations, summary over it >= discard): for every scenario x sensor the 15 validation atmospheres
(5 LOS r0 x 3 wind/profile draws) are run with every controller on the same phase screens, so the comparison is paired per atmosphere:
  rel_ref(c, atm) = WFE(c, atm) / WFE(ref, atm) - 1            ref = POL-ZOH (the predictor's own baseline) and the integrator
per scenario x sensor x controller: median and IQR of the WFE, median of rel with a bootstrap 95 % CI, win fraction (rel < 0), Wilcoxon
signed-rank test (two-sided, Holm correction over all the comparisons of the table) and diverged runs. The same statistics without the
atmospheres of the gain scan (atm1/3/5 draw1, used to choose the baseline gains) as a sensitivity check.
Trend: rel vs the coherence time tau0 = 0.314 r0_LOS / v_eff (v_eff = (sum f_i v_i^5/3)^3/5, scaled by the wind factor) at 500 nm.

Open loop (polPredictionErrors, 5 atmospheres, draw 1): prediction error rms [px] with OL input (pure prediction) and POL input (prediction
+ POL reconstruction), skill = 1 - err / err_ZOH.

Scenario of a run from its config: S1 2 kHz no vibrations, S2 vibrations, S3 vibrations + wind x2, S4 vibrations + wind x0.75 at 700 Hz,
S5 no vibrations + WFS noise. The predictors were trained only on 36x36, 2 kHz, no vibrations, no noise (generateTrainingData.py):
S2-S5 and every 50x50 run are out of the training distribution.

Output in <results>/analysis: cl_runs.csv (one row per run), cl_stats.csv, cl_stats_noScanAtm.csv, ol_stats.csv, table_cl.tex and the
figures cl_relative.png/pdf, cl_vs_tau0.png/pdf, ol_skill.png/pdf.

Usage: WFEPRED_CAMPAIGN=v3 python analyzePublicationCampaign.py [--base_dir ~/simulations]
       WFEPRED_CAMPAIGN=v3 python analyzePublicationCampaign.py --pol-gain S1_50=0.5 S2_50=0.5 S3_50=0.5 --out-tag _polg0.5_50
"""
import os
import sys
import glob
import json
import argparse
import numpy as np
import pandas as pd
import h5py
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common                                     # noqa: E402
from atmosphereCases import atm_cases             # noqa: E402

SCENARIOS = {'S1': '2 kHz', 'S2': '2 kHz + vib.', 'S3': '2 kHz + vib. + wind ×2', 'S4': '700 Hz + vib. + wind ×0.75', 'S5': '2 kHz + WFS noise'}
CASES = ['integrator', 'pol:zoh', 'pol:linear', 'pol:incAR', 'pol:lstm', 'pol:lstm_mse']
LABELS = {'integrator': 'Integrator', 'pol:zoh': 'POL-ZOH', 'pol:linear': 'POL-linear', 'pol:incAR': 'POL-incAR',
          'pol:lstm': 'POL-LSTM (band)', 'pol:lstm_mse': 'POL-LSTM (MSE)'}
COLORS = {'integrator': '#7f7f7f', 'pol:zoh': '#222222', 'pol:linear': '#e3a21a', 'pol:incAR': '#1baf7a', 'pol:lstm': '#2a78d6',
          'pol:lstm_mse': '#d55181'}
SCAN = {('atm1', 'draw1'), ('atm3', 'draw1'), ('atm5', 'draw1')}
DIVERGED_NM = 1000     # WFE above this counts as diverged (the flag of closedLoopComparison misses slow divergences)
LOSS = 1e6             # relative change [%] given to a diverged run in the rank statistics (only its rank matters)


def short_case(c):
    if c.startswith('pol:incAR'):
        return 'pol:incAR'
    if c.startswith('pol:lstm'):
        return 'pol:lstm_mse' if c.endswith('_mse') else 'pol:lstm'
    return c


def scenario(cfg):
    if cfg.get('wfs_noise'):
        return 'S5'
    if cfg['fs'] != 2000:
        return 'S4'
    if not cfg['vibrations']:
        return 'S1'
    return 'S3' if cfg['wind_scale'] > 1 else 'S2'


def tau0(atm, draw, wind_scale):
    kw = atm_cases[atm][draw]
    f, v = np.asarray(kw['fractionalR0']), wind_scale * np.asarray(kw['windSpeed'])
    v_eff = np.sum(f / f.sum() * v ** (5 / 3)) ** (3 / 5)
    return 0.314 * kw['los_r0'] / v_eff, kw['los_r0'], v_eff


def select_pol_gain(df, choice):
    """Keep, for every scenario x sensor, the POL runs of one POL gain: the one in choice ({'S1_50': 0.5}) or else the most frequent.
    The integrator does not depend on the POL gain: one run per atmosphere."""
    keep = []
    for (sc, s), g in df.groupby(['scenario', 'sensor']):
        pol = g[g.case != 'integrator']
        gp = choice.get(f'{sc}_{s}', pol.pol_gain.mode().iloc[0])
        keep.append(pol[np.isclose(pol.pol_gain, gp)])
        keep.append(g[g.case == 'integrator'].drop_duplicates(['atm', 'draw']))
    return pd.concat(keep, ignore_index=True)


def load_cl(root):
    rows = []
    for d in sorted(glob.glob(os.path.join(root, 'closedLoopComparison', '*_1000it_*'))):
        if d.endswith('_test'):
            continue
        for p in glob.glob(os.path.join(d, 'res_*.h5')):
            with h5py.File(p, 'r') as f:
                cfg, sm = json.loads(f.attrs['config']), json.loads(f.attrs['summary'])
            t0, r0, v = tau0(cfg['atm'], cfg['draw'], cfg['wind_scale'])
            div = bool(sm['diverged']) or not np.isfinite(sm['wfe_nm']) or sm['wfe_nm'] > DIVERGED_NM
            rows.append(dict(scenario=scenario(cfg), sensor=cfg['sensor'], atm=cfg['atm'], draw=cfg['draw'], case=short_case(sm['case']),
                             gain=sm['gain'], pol_gain=cfg['gain'], wfe_nm=np.inf if div else sm['wfe_nm'], diverged=div,
                             los_r0_cm=100 * r0, v_eff=v, tau0_ms=1e3 * t0, scan_atm=(cfg['atm'], cfg['draw']) in SCAN))
    return pd.DataFrame(rows)


def boot_median_ci(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    m = np.median(rng.choice(x, (n, len(x)), replace=True), axis=1)
    return np.percentile(m, [2.5, 97.5])


def holm(p):
    p = np.asarray(p, float)
    order, out, run = np.argsort(p), np.empty_like(p), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - k) * p[i]))
        out[i] = run
    return out


def cl_stats(df):
    wide = df.pivot_table(index=['scenario', 'sensor', 'atm', 'draw'], columns='case', values='wfe_nm', aggfunc='first')
    rows = []
    for (sc, s), g in wide.groupby(level=['scenario', 'sensor']):
        for c in CASES:
            if c not in g:
                continue
            w = g[c].to_numpy()
            r = dict(scenario=sc, sensor=s, case=c, n=len(w), diverged=int(np.sum(~np.isfinite(w))), gain=df[(df.scenario == sc) & (df.sensor == s) & (df.case == c)].gain.iloc[0],
                     wfe_median=np.median(w), wfe_q25=np.percentile(w, 25), wfe_q75=np.percentile(w, 75), wfe_mean=np.mean(w[np.isfinite(w)]))
            for ref, tag in (('pol:zoh', 'zoh'), ('integrator', 'int')):
                if c == ref or ref not in g:
                    continue
                wr = g[ref].to_numpy()
                # paired change over all the atmospheres: a diverged run loses (LOSS), a diverged reference counts as a win (-100 %)
                rel = np.where(np.isfinite(w) & np.isfinite(wr), 100 * (w / wr - 1), np.where(np.isfinite(w), -100.0, LOSS))
                lo, hi = boot_median_ci(rel)
                fin = rel[np.abs(rel) < LOSS]
                r.update({f'rel_{tag}_median_pct': np.median(rel), f'rel_{tag}_ci_lo': lo, f'rel_{tag}_ci_hi': hi,
                          f'rel_{tag}_mean_pct': np.mean(fin) if len(fin) else np.nan, f'win_{tag}': np.mean(rel < 0),
                          f'p_{tag}': wilcoxon(rel).pvalue if np.any(rel != 0) else 1.0})
            rows.append(r)
    st = pd.DataFrame(rows)
    for tag in ('zoh', 'int'):
        m = st[f'p_{tag}'].notna()
        st.loc[m, f'p_{tag}_holm'] = holm(st.loc[m, f'p_{tag}'])
    return st


def load_ol(root):
    rows = []
    for p in sorted(glob.glob(os.path.join(root, 'pol_prediction_errors', 'res_polpred_*.h5'))):
        with h5py.File(p, 'r') as f:
            cfg, sm = json.loads(f.attrs['config']), json.loads(f.attrs['summary'])
        cfg.setdefault('wfs_noise', None)
        for pred in cfg['predictors']:
            rows.append(dict(scenario=scenario(cfg), sensor=cfg['sensor'], atm=cfg['atm'], case=short_case('pol:' + pred), ol_rms_px=sm['ol_rms_px'],
                             pol_err_px=sm['pol_err_px'], err_ol_px=sm[f'pred_err_ol_{pred}_px'], err_pol_px=sm[f'pred_err_pol_{pred}_px']))
    df = pd.DataFrame(rows)
    zoh = df[df.case == 'pol:zoh'].set_index(['scenario', 'sensor', 'atm'])
    k = df.set_index(['scenario', 'sensor', 'atm']).index
    df['skill_ol'] = 1 - df.err_ol_px.to_numpy() / zoh.loc[k, 'err_ol_px'].to_numpy()
    df['skill_pol'] = 1 - df.err_pol_px.to_numpy() / zoh.loc[k, 'err_pol_px'].to_numpy()
    return df


def plot_relative(df, out):
    wide = df.pivot_table(index=['scenario', 'sensor', 'atm', 'draw'], columns='case', values='wfe_nm', aggfunc='first')
    cases = [c for c in CASES if c != 'pol:zoh']
    fig, axes = plt.subplots(2, 5, figsize=(17, 7.5), sharey=True)
    for i, s in enumerate((36, 50)):
        for j, sc in enumerate(SCENARIOS):
            ax = axes[i, j]
            g = wide.xs((sc, s), level=('scenario', 'sensor'))
            for k, c in enumerate(cases):
                rel = 100 * (g[c] / g['pol:zoh'] - 1).to_numpy()
                n_div = int(np.sum(~np.isfinite(rel)))
                rel = rel[np.isfinite(rel)]
                ax.boxplot(rel, positions=[k], widths=0.6, showfliers=False, patch_artist=True,
                           boxprops=dict(facecolor=COLORS[c], alpha=0.35, edgecolor=COLORS[c]), medianprops=dict(color=COLORS[c], lw=2),
                           whiskerprops=dict(color=COLORS[c]), capprops=dict(color=COLORS[c]))
                ax.plot(k + np.random.default_rng(k).uniform(-0.18, 0.18, len(rel)), rel, '.', color=COLORS[c], ms=4)
                if n_div:
                    ax.text(k, 0.97, f'{n_div} div.', transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=7, color='#c0392b')
            ax.axhline(0, color='k', lw=0.8)
            ax.set_xticks(range(len(cases)), [LABELS[c] for c in cases], rotation=45, ha='right', fontsize=8)
            ax.set_title(f'{sc} {s}x{s}: {SCENARIOS[sc]}', fontsize=9)
            ax.grid(True, axis='y', color='0.92')
            ax.spines[['right', 'top']].set_visible(False)
        axes[i, 0].set_ylabel('WFE / WFE(POL-ZOH) − 1 [%]')
    lim = np.nanpercentile(np.abs(100 * (wide.drop(columns='pol:zoh').to_numpy() / wide[['pol:zoh']].to_numpy() - 1))[np.isfinite(wide.drop(columns='pol:zoh').to_numpy())], 99)
    axes[0, 0].set_ylim(-min(lim, 40), min(lim, 60))
    fig.suptitle('Closed loop, paired by atmosphere (15 atmospheres per panel): change of the residual WFE relative to POL-ZOH (negative = better)', fontsize=11)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(f'{out}.{ext}', dpi=160)
    plt.close(fig)


def plot_vs_tau0(df, out):
    wide = df.pivot_table(index=['scenario', 'sensor', 'atm', 'draw'], columns='case', values='wfe_nm', aggfunc='first')
    tau = df.drop_duplicates(['scenario', 'sensor', 'atm', 'draw']).set_index(['scenario', 'sensor', 'atm', 'draw']).tau0_ms
    fig, axes = plt.subplots(2, 5, figsize=(17, 7), sharey=True)
    for i, s in enumerate((36, 50)):
        for j, sc in enumerate(SCENARIOS):
            ax = axes[i, j]
            g = wide.xs((sc, s), level=('scenario', 'sensor'), drop_level=False)
            t = tau.loc[g.index].to_numpy()
            for c in ('integrator', 'pol:incAR', 'pol:lstm', 'pol:lstm_mse'):
                rel = 100 * (g[c] / g['pol:zoh'] - 1).to_numpy()
                ok = np.isfinite(rel)
                ax.plot(t[ok], rel[ok], 'o', color=COLORS[c], ms=4, label=LABELS[c])
                if ok.sum() > 3:
                    a, b = np.polyfit(np.log(t[ok]), rel[ok], 1)
                    tt = np.linspace(t[ok].min(), t[ok].max(), 20)
                    ax.plot(tt, a * np.log(tt) + b, '-', color=COLORS[c], lw=1)
            ax.axhline(0, color='k', lw=0.8)
            ax.set_xscale('log')
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
            ax.xaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
            ax.set_title(f'{sc} {s}x{s}', fontsize=9)
            ax.set_xlabel('τ0 at 500 nm [ms]')
            ax.grid(True, color='0.92')
            ax.spines[['right', 'top']].set_visible(False)
        axes[i, 0].set_ylabel('WFE / WFE(POL-ZOH) − 1 [%]')
    axes[0, 0].legend(fontsize=7, frameon=False)
    axes[0, 0].set_ylim(-30, 40)
    fig.suptitle('Closed loop: relative change vs the coherence time of the atmosphere (lines: fit in log τ0)', fontsize=11)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(f'{out}.{ext}', dpi=160)
    plt.close(fig)


def plot_ol(ol, out):
    cases = ['pol:linear', 'pol:incAR', 'pol:lstm', 'pol:lstm_mse']
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), sharey=True)
    for ax, key, title in ((axes[0], 'skill_ol', 'OL input (pure prediction)'), (axes[1], 'skill_pol', 'POL input (prediction + POL reconstruction)')):
        x = 0
        ticks = []
        for sc in SCENARIOS:
            for s in (36, 50):
                for k, c in enumerate(cases):
                    v = 100 * ol[(ol.scenario == sc) & (ol.sensor == s) & (ol.case == c)][key].to_numpy()
                    if len(v):
                        ax.bar(x + 0.2 * k, np.median(v), 0.2, color=COLORS[c], label=LABELS[c] if (sc, s) == ('S1', 36) else None)
                        ax.plot(np.full(len(v), x + 0.2 * k), v, '.', color='k', ms=2)
                ticks.append((x + 0.3, f'{sc}\n{s}'))
                x += 1.2
            x += 0.4
        ax.axhline(0, color='k', lw=0.8)
        ax.set_xticks([t[0] for t in ticks], [t[1] for t in ticks], fontsize=7)
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis='y', color='0.92')
        ax.spines[['right', 'top']].set_visible(False)
    axes[0].set_ylabel('Skill vs ZOH: 1 − err / err_ZOH [%]')
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle('Open-loop prediction error (h = 2 samples), median of 5 atmospheres (dots: each atmosphere)', fontsize=11)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(f'{out}.{ext}', dpi=160)
    plt.close(fig)


def latex_table(st, path):
    lines = [r'\begin{tabular}{llrrrrr}', r'\hline',
             r'Scenario & Controller & WFE [nm] & $\Delta$ vs POL-ZOH [\%] & 95\% CI & wins & $p_\mathrm{Holm}$ \\', r'\hline']
    for (sc, s), g in st.groupby(['scenario', 'sensor']):
        for _, r in g.iterrows():
            if r.case == 'pol:zoh':
                d = ci = w = p = '--'
            else:
                fmt = lambda v: 'div.' if v >= LOSS else f'{v:+.1f}'
                d, ci = fmt(r.rel_zoh_median_pct), f'[{fmt(r.rel_zoh_ci_lo)}, {fmt(r.rel_zoh_ci_hi)}]'
                w, p = f'{round(15 * r.win_zoh):d}/15', f'{r.p_zoh_holm:.2g}'
            div = f' ({r.diverged} div.)' if r.diverged else ''
            wfe = 'div.' if not np.isfinite(r.wfe_median) else f'{r.wfe_median:.1f}'
            lines.append(f'{sc} {s}$\\times${s} & {LABELS[r.case]} & {wfe}{div} & {d} & {ci} & {w} & {p} \\\\')
        lines.append(r'\hline')
    lines.append(r'\end{tabular}')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Statistics of the publication campaign')
    parser.add_argument('--base_dir', default=os.path.expanduser('~/simulations'))
    parser.add_argument('--pol-gain', nargs='+', default=[], help='POL gain per scenario_sensor, e.g. S1_50=0.5 (default: the most frequent)')
    parser.add_argument('--out-tag', default='', help='Suffix of the output folder analysis<tag>')
    args = parser.parse_args()
    root = common.results_root(args.base_dir)
    out = os.path.join(root, 'analysis' + args.out_tag)
    choice = {k: float(v) for k, v in (x.split('=') for x in args.pol_gain)}
    os.makedirs(out, exist_ok=True)

    df = select_pol_gain(load_cl(root), choice)
    print('POL gain per scenario x sensor: ' + ', '.join(f'{sc}_{s} {g.pol_gain.iloc[0]:g}' for (sc, s), g in df[df.case != 'integrator'].groupby(['scenario', 'sensor'])))
    df.to_csv(os.path.join(out, 'cl_runs.csv'), index=False)
    n = df.groupby(['scenario', 'sensor', 'case']).size()
    print(f'Closed loop: {len(df)} runs, {n.min()}-{n.max()} atmospheres per scenario x sensor x controller')
    st = cl_stats(df)
    st.to_csv(os.path.join(out, 'cl_stats.csv'), index=False, float_format='%.5g')
    st_ns = cl_stats(df[~df.scan_atm])
    st_ns.to_csv(os.path.join(out, 'cl_stats_noScanAtm.csv'), index=False, float_format='%.5g')
    latex_table(st, os.path.join(out, 'table_cl.tex'))
    pd.set_option('display.width', 250)
    cols = ['scenario', 'sensor', 'case', 'gain', 'diverged', 'wfe_median', 'rel_zoh_median_pct', 'rel_zoh_ci_lo', 'rel_zoh_ci_hi', 'win_zoh', 'p_zoh_holm',
            'rel_int_median_pct', 'win_int', 'p_int_holm']
    print(st[cols].to_string(index=False, float_format=lambda v: f'{v:.3g}'))
    print('\nWithout the gain-scan atmospheres (12 per panel): median rel vs POL-ZOH [%]')
    print(st_ns.pivot_table(index=['scenario', 'sensor'], columns='case', values='rel_zoh_median_pct').round(1).to_string())

    ol = load_ol(root)
    ol.to_csv(os.path.join(out, 'ol_runs.csv'), index=False)
    ols = ol.groupby(['scenario', 'sensor', 'case']).agg(err_ol_px=('err_ol_px', 'median'), err_pol_px=('err_pol_px', 'median'), skill_ol=('skill_ol', 'median'),
                                                         skill_pol=('skill_pol', 'median'), pol_err_px=('pol_err_px', 'median'), n=('atm', 'size')).reset_index()
    ols.to_csv(os.path.join(out, 'ol_stats.csv'), index=False, float_format='%.5g')
    print('\nOpen loop: median skill vs ZOH [%] (OL input / POL input)')
    print((100 * ols.pivot_table(index=['scenario', 'sensor'], columns='case', values=['skill_ol', 'skill_pol'])).round(1).to_string())

    plot_relative(df, os.path.join(out, 'cl_relative'))
    plot_vs_tau0(df, os.path.join(out, 'cl_vs_tau0'))
    plot_ol(ol, os.path.join(out, 'ol_skill'))
    print(f'\nSaved in {out}')


if __name__ == '__main__':
    main()
