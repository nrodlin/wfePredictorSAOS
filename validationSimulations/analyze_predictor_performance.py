#!/usr/bin/env python3
"""
AO Slope Predictor Performance Analysis Toolkit
================================================
Comprehensive analysis of 2 kHz AO Slope Predictor simulations:
1. Open-Loop (OL) Prediction Error:
   - Compares: no prediction (ZOH 2-sample delay baseline), linear extrapolation and LSTM.
   - Computes MAE, RMSE and the RMSE reduction (%) w.r.t. ZOH.
2. Pseudo-Open Loop Prediction Error (CL sin cerrar):
   - Evaluates the prediction error on the reconstructed POL slopes, after the loop transient.
3. Closed-Loop (CL) Performance:
   - Every results folder 'cl_*' (except cl_sin_cerrar) is a configuration, compared against a baseline folder.
   - Computes Strehl Ratio (56 Hz & 5 Hz), Solar Contrast and WFE RMS after the loop transient.
   - Incomplete runs (fewer iterations than expected) are excluded and listed in the report.

Single improvement metric: RMSE reduction [%] = (RMSE_ZOH - RMSE_pred) / RMSE_ZOH, computed from aggregated RMSEs
(RMS of the per-case RMSEs), never as an average of per-case percentages.
36x36 and 50x50 sensors are always reported separately (different band, plate scale and DM).
"""

import re
import argparse
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import h5py
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# The predictors of redArmSolarSCAO_04_CL_sin_cerrar.py need past_horizon=24 samples: the first stored sample is iteration 23
SIN_CERRAR_FIRST_ITERATION = 23

# Filename tail shared by all the validation scripts: <sensor>_<vibr>_<atm>_<draw>.h5
CASE_PATTERN = re.compile(r'(\d+x\d+)_(noVibr|Vibr)_(atm\d+)_(draw\d+)\.(h5|npy)$')

# Fixed categorical order for the closed-loop configurations (baseline is always neutral gray)
BASELINE_COLOR = '#7f7f7f'
CONFIG_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#8f6fd6', '#c44e3b']

CL_METRICS = ['Strehl_56Hz', 'Strehl_5Hz', 'Contrast_Solar', 'WFE_RMS_nm']


class PredictorPerformanceAnalyzer:
    def __init__(self, base_dir=None, ol_dir=None, output_dir=None, baseline=None, n_iterations=2500,
                 sampling_freq=2000.0, discard_time=0.2, logger=None):
        self.logger = logger or self._init_logger()
        self.base_dir = Path(base_dir) if base_dir else self._resolve_base_dir()
        self.ol_dir = Path(ol_dir) if ol_dir else self._resolve_ol_dir()
        self.output_dir = Path(output_dir) if output_dir else (self.base_dir / "analysis_results")
        self.baseline = baseline
        self.n_iterations = int(n_iterations)
        self.discard_its = int(round(discard_time * sampling_freq))
        self.discard_time = discard_time

        # Runs excluded from the analysis, reported at the end: list of (file, reason)
        self.excluded = []

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "figures").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "reports").mkdir(parents=True, exist_ok=True)

        self.logger.info("=" * 80)
        self.logger.info(" AO PREDICTOR PERFORMANCE ANALYZER")
        self.logger.info(f" CL Base Directory: {self.base_dir}")
        self.logger.info(f" OL Base Directory: {self.ol_dir}")
        self.logger.info(f" Output Directory:  {self.output_dir}")
        self.logger.info(f" Expected iterations: {self.n_iterations} | Transient discarded: {self.discard_its} its ({discard_time} s)")
        self.logger.info("=" * 80)

    def _init_logger(self):
        logger = logging.getLogger("PredictorAnalyzer")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            ch = logging.StreamHandler()
            fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", datefmt="%H:%M:%S")
            ch.setFormatter(fmt)
            logger.addHandler(ch)
        return logger

    def _resolve_base_dir(self):
        candidates = [
            Path.home() / "simulations" / "results" / "thirdAttempt",
            Path("/net/durum/scratch") / Path.home().name / "simulations" / "results" / "thirdAttempt",
        ]
        for c in candidates:
            if c.exists():
                return c
        return candidates[0]

    def _resolve_ol_dir(self):
        candidates = [
            self.base_dir / "predictor_ol",
            Path.home() / "simulations" / "results" / "predictor_ol",
        ]
        for c in candidates:
            if c.exists():
                return c
        return candidates[0]

    def _exclude(self, file, reason):
        self.excluded.append((str(file), reason))
        self.logger.warning(f"Excluded {Path(file).name}: {reason}")

    @staticmethod
    def _parse_case(filename):
        m = CASE_PATTERN.search(filename)
        if m is None:
            return None
        return {'sensor': m.group(1), 'vibr': m.group(2), 'atm': m.group(3), 'draw': m.group(4)}

    @staticmethod
    def _rmse_improvement(rmse_ref, rmse_pred):
        return (rmse_ref - rmse_pred) / rmse_ref * 100.0

    # ==============================================================================
    # 1-2. PREDICTION ERRORS (OPEN LOOP & POL SIN CERRAR)
    # ==============================================================================

    def _error_row(self, domain, case, truth, zoh, plin, plstm):
        row = {'domain': domain, **case, 'n_samples': len(truth)}
        for name, pred in [('Baseline_ZOH', zoh), ('Linear', plin), ('LSTM', plstm)]:
            if pred is None:
                row[f'MAE_{name}'] = np.nan
                row[f'RMSE_{name}'] = np.nan
            else:
                row[f'MAE_{name}'] = float(np.mean(np.abs(pred - truth)))
                row[f'RMSE_{name}'] = float(np.sqrt(np.mean((pred - truth) ** 2)))
        row['Impr_RMSE_Linear_pct'] = self._rmse_improvement(row['RMSE_Baseline_ZOH'], row['RMSE_Linear'])
        row['Impr_RMSE_LSTM_pct'] = self._rmse_improvement(row['RMSE_Baseline_ZOH'], row['RMSE_LSTM'])
        return row

    def analyze_open_loop(self):
        """
        Calculates MAE, RMSE and RMSE reduction over ZOH for all Open-Loop cases (no transient in open loop).
        """
        self.logger.info(">>> Analyzing Open-Loop Prediction Errors...")
        rows = []

        if not self.ol_dir.exists():
            self.logger.warning(f"Open Loop directory not found: {self.ol_dir}")
            return pd.DataFrame()

        for truth_p in sorted(self.ol_dir.glob("truth_*.npy")):
            case = self._parse_case(truth_p.name)
            if case is None:
                self._exclude(truth_p, "unrecognised file name")
                continue
            key = truth_p.name[len("truth_"):]
            paths = {k: self.ol_dir / f"{k}_{key}" for k in ['delayed', 'pred_linear', 'pred_lstm']}
            missing = [k for k, p in paths.items() if not p.exists()]
            if missing:
                self._exclude(truth_p, f"missing {missing}")
                continue

            truth = np.load(truth_p)
            zoh, plin, plstm = [np.load(paths[k]) for k in ['delayed', 'pred_linear', 'pred_lstm']]
            rows.append(self._error_row('Open Loop', case, truth, zoh, plin, plstm))

        df_ol = pd.DataFrame(rows)
        self.logger.info(f"Open-Loop dataset loaded: {len(df_ol)} cases processed.")
        return df_ol

    def analyze_cl_sin_cerrar(self):
        """
        Calculates MAE, RMSE and RMSE reduction for the predictors on the POL slopes (CL sin cerrar).
        The arrays start at iteration SIN_CERRAR_FIRST_ITERATION; the loop transient is discarded.
        """
        self.logger.info(">>> Analyzing Pseudo-Open Loop (CL sin cerrar) Prediction Errors...")
        rows = []
        sc_dir = self.base_dir / "cl_sin_cerrar"

        if not sc_dir.exists():
            self.logger.warning(f"cl_sin_cerrar directory not found: {sc_dir}")
            return pd.DataFrame()

        expected_len = self.n_iterations - SIN_CERRAR_FIRST_ITERATION
        first = max(0, self.discard_its - SIN_CERRAR_FIRST_ITERATION)

        for pol_p in sorted(sc_dir.glob("slopes_pol_*.npy")):
            case = self._parse_case(pol_p.name)
            if case is None:
                self._exclude(pol_p, "unrecognised file name")
                continue
            key = pol_p.name[len("slopes_pol_"):]
            plin_p = sc_dir / f"pred_pol_linear_{key}"
            plstm_p = sc_dir / f"pred_pol_lstm_{key}"
            if not (plin_p.exists() and plstm_p.exists()):
                self._exclude(pol_p, "missing predictor arrays")
                continue

            pol_arr = np.load(pol_p)
            if len(pol_arr) < expected_len:
                self._exclude(pol_p, f"incomplete run ({len(pol_arr)} of {expected_len} samples)")
                continue

            pred_lin = np.load(plin_p)
            pred_lstm = np.load(plstm_p)

            # Predictors target the POL slopes 2 steps ahead: prediction made at k is compared with pol[k+2]
            target_pol = pol_arr[first + 2:]
            eval_zoh = pol_arr[first:-2]
            rows.append(self._error_row('POL sin cerrar', case, target_pol, eval_zoh, pred_lin[first:-2], pred_lstm[first:-2]))

        df_sc = pd.DataFrame(rows)
        self.logger.info(f"POL sin cerrar dataset loaded: {len(df_sc)} cases processed.")
        return df_sc

    @staticmethod
    def aggregate_errors(df, keys):
        """
        Aggregates prediction errors per group: RMSE as RMS of the per-case RMSEs (pooled MSE), MAE as mean,
        and the RMSE reduction recomputed from the aggregated RMSEs.
        """
        if df.empty:
            return df
        rms = lambda x: float(np.sqrt(np.mean(np.square(x))))
        agg_spec = {'n_cases': ('RMSE_Baseline_ZOH', 'size')}
        for name in ['Baseline_ZOH', 'Linear', 'LSTM']:
            agg_spec[f'RMSE_{name}'] = (f'RMSE_{name}', rms)
            agg_spec[f'MAE_{name}'] = (f'MAE_{name}', 'mean')
        agg = df.groupby(keys).agg(**agg_spec).reset_index()
        agg['Impr_RMSE_Linear_pct'] = (agg['RMSE_Baseline_ZOH'] - agg['RMSE_Linear']) / agg['RMSE_Baseline_ZOH'] * 100.0
        agg['Impr_RMSE_LSTM_pct'] = (agg['RMSE_Baseline_ZOH'] - agg['RMSE_LSTM']) / agg['RMSE_Baseline_ZOH'] * 100.0
        return agg

    # ==============================================================================
    # 3. CLOSED-LOOP PERFORMANCE ANALYSIS
    # ==============================================================================

    def _closed_loop_configs(self):
        configs = sorted(d.name for d in self.base_dir.iterdir() if d.is_dir() and d.name.startswith('cl_') and d.name != 'cl_sin_cerrar')
        if self.baseline is None:
            candidates = [c for c in configs if c.startswith('cl_baseline')]
            if len(candidates) == 0:
                self.logger.warning("No baseline folder found (cl_baseline*), relative gains will not be computed")
            else:
                preferred = [c for c in ['cl_baseline', 'cl_baseline_gain0.25'] if c in candidates]
                self.baseline = preferred[0] if preferred else candidates[0]
                if len(candidates) > 1:
                    self.logger.warning(f"Several baselines found {candidates}, using {self.baseline} (use --baseline to choose)")
        elif self.baseline not in configs:
            raise ValueError(f"Baseline folder {self.baseline} not found in {self.base_dir}")
        return configs

    def _mean_after_transient(self, grp, key, squared=False):
        """
        Mean of grp[key] over the samples fully acquired after the transient. Each sample covers the iterations
        (previous saved iteration, own iteration], so long exposures overlapping the transient are discarded.
        """
        values = grp[key][:].ravel()
        iterations = grp['iteration'][:].ravel()
        starts = np.concatenate(([0], iterations[:-1] + 1))
        keep = starts >= self.discard_its
        if not np.any(keep):
            return np.nan
        if squared:
            return float(np.sqrt(np.mean(values[keep] ** 2)))
        return float(np.mean(values[keep]))

    @staticmethod
    def _last_iteration(hf):
        last = [-1]
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and name.endswith('/iteration') and obj.shape[0] > 0:
                last.append(int(obj[-1]))
        hf.visititems(visit)
        return max(last)

    def analyze_closed_loop(self):
        """
        Extracts Strehl Ratio (56Hz, 5Hz), Contrast and WFE RMS for all complete closed-loop runs and computes
        the relative change of each configuration against the baseline folder.
        """
        self.logger.info(">>> Analyzing Closed-Loop Runs...")
        configs = self._closed_loop_configs()
        rows = []

        for cfg in configs:
            for fp in sorted((self.base_dir / cfg).glob("*.h5")):
                case = self._parse_case(fp.name)
                if case is None:
                    self._exclude(fp, "unrecognised file name")
                    continue
                try:
                    with h5py.File(fp, 'r') as hf:
                        last_it = self._last_iteration(hf)
                        if last_it < self.n_iterations - 1:
                            self._exclude(fp, f"incomplete run (last iteration {last_it + 1} of {self.n_iterations})")
                            continue

                        row = {'config': cfg, **case}
                        # Strehl 56 Hz (LightPath_1) and 5 Hz (LightPath_2), Solar contrast (LightPath_0)
                        for metric, lp, field in [('Strehl_56Hz', 'LightPath_1', 'strehl'), ('Strehl_5Hz', 'LightPath_2', 'strehl'),
                                                  ('Contrast_Solar', 'LightPath_0', 'contrast')]:
                            path = f'{lp}/sci_frame_longExp'
                            row[metric] = self._mean_after_transient(hf[path], field) if (path in hf and field in hf[path]) else np.nan
                        # WFE: temporal RMS of the per-iteration spatial std of the science OPD [nm]
                        path = 'LightPath_1/sci_opd'
                        row['WFE_RMS_nm'] = self._mean_after_transient(hf[path], 'std', squared=True) * 1e9 if (path in hf and 'std' in hf[path]) else np.nan
                        rows.append(row)
                except Exception as e:
                    self._exclude(fp, f"read error: {e}")

        df_cl_raw = pd.DataFrame(rows)
        if df_cl_raw.empty:
            return pd.DataFrame(), pd.DataFrame()

        keys = ['sensor', 'vibr', 'atm', 'draw']
        duplicated = df_cl_raw[df_cl_raw.duplicated(subset=['config'] + keys, keep=False)]
        if not duplicated.empty:
            raise ValueError(f"Several runs for the same case in a folder (e.g. different delays), separate them:\n{duplicated[['config'] + keys]}")

        piv = df_cl_raw.pivot(index=keys, columns='config', values=CL_METRICS)
        piv.columns = [f'{metric}_{cfg}' for metric, cfg in piv.columns]
        piv = self.add_cl_gains(piv.reset_index(), configs, self.baseline)

        self.logger.info(f"Closed-Loop dataset processed: {len(df_cl_raw)} complete runs across {len(piv)} unique conditions.")
        return df_cl_raw, piv

    @staticmethod
    def add_cl_gains(df, configs, baseline):
        """Relative change of every configuration against the baseline. Recomputed after any aggregation."""
        if baseline is None:
            return df
        for c in configs:
            if c == baseline:
                continue
            for metric, sign, name in [('Strehl_56Hz', 1, 'Gain_SR_56Hz'), ('Strehl_5Hz', 1, 'Gain_SR_5Hz'),
                                       ('Contrast_Solar', 1, 'Gain_Contrast'), ('WFE_RMS_nm', -1, 'Reduction_WFE')]:
                col, ref = f'{metric}_{c}', f'{metric}_{baseline}'
                if col in df.columns and ref in df.columns:
                    df[f'{name}_{c}_pct'] = sign * (df[col] - df[ref]) / df[ref] * 100.0
        return df

    def aggregate_cl(self, df_cl_piv, keys, configs):
        metric_cols = [c for c in df_cl_piv.columns if any(c.startswith(m + '_') for m in CL_METRICS)]
        agg = df_cl_piv.groupby(keys)[metric_cols].mean().reset_index()
        return self.add_cl_gains(agg, configs, self.baseline)

    # ==============================================================================
    # 4. FIGURES (one per sensor)
    # ==============================================================================

    def _save(self, fig, name):
        out_p = self.output_dir / "figures" / name
        fig.savefig(out_p, dpi=300, bbox_inches='tight')
        fig.savefig(out_p.with_suffix('.pdf'), bbox_inches='tight')
        self.logger.info(f"  [SAVED] {out_p.name}")
        plt.close(fig)

    def plot_prediction_errors_comparison(self, df_ol, df_sc, sensor):
        """
        Plots prediction RMSE and RMSE reduction for Open Loop vs POL sin cerrar, for one sensor.
        """
        datasets = [(name, df[df['sensor'] == sensor]) for name, df in [('Open Loop', df_ol), ('POL sin cerrar', df_sc)] if not df.empty]
        datasets = [(name, df) for name, df in datasets if not df.empty]
        if not datasets:
            return None

        fig, axes = plt.subplots(len(datasets), 2, figsize=(16, 5.5 * len(datasets)), squeeze=False)
        fig.subplots_adjust(top=0.90, bottom=0.10, hspace=0.30, wspace=0.22)

        styles = {'Baseline_ZOH': ('#555555', '--', 'o', 'ZOH'), 'Linear': ('#2a78d6', '-', 's', 'Linear'), 'LSTM': ('#eb6834', '-', '^', 'LSTM')}
        for (name, df), (ax_rmse, ax_impr) in zip(datasets, axes):
            agg = self.aggregate_errors(df, ['atm', 'vibr'])
            atms = sorted(agg['atm'].unique(), key=lambda a: int(a[3:]))
            x = np.arange(len(atms))

            for vibr, ls_vib, alpha in [('noVibr', None, 1.0), ('Vibr', ':', 0.7)]:
                sub = agg[agg['vibr'] == vibr].set_index('atm').reindex(atms)
                if sub['RMSE_Baseline_ZOH'].isna().all():
                    continue
                for col, (color, ls, marker, label) in styles.items():
                    ax_rmse.plot(x, sub[f'RMSE_{col}'], color=color, linestyle=ls_vib or ls, marker=marker, alpha=alpha, linewidth=1.8, label=f'{label} ({vibr})')

            ax_rmse.set_xticks(x)
            ax_rmse.set_xticklabels(atms)
            ax_rmse.set_title(f"{name} {sensor}: slope prediction RMSE [px]", fontsize=12, fontweight='bold')
            ax_rmse.set_xlabel("Atmosphere (turbulence level)")
            ax_rmse.set_ylabel("RMSE [px]")
            ax_rmse.grid(True, linestyle=":", alpha=0.5)
            ax_rmse.legend(fontsize=8, loc='upper right')

            width = 0.2
            bars = [('noVibr', 'Linear', '#2a78d6', None), ('noVibr', 'LSTM', '#eb6834', None), ('Vibr', 'Linear', '#2a78d6', '//'), ('Vibr', 'LSTM', '#eb6834', '//')]
            for k, (vibr, col, color, hatch) in enumerate(bars):
                sub = agg[agg['vibr'] == vibr].set_index('atm').reindex(atms)
                if sub[f'Impr_RMSE_{col}_pct'].isna().all():
                    continue
                ax_impr.bar(x + (k - 1.5) * width, sub[f'Impr_RMSE_{col}_pct'], width, label=f'{col} ({vibr})', color=color, hatch=hatch, edgecolor='white', linewidth=0.5)
            ax_impr.axhline(0, color='black', linewidth=0.8)
            ax_impr.set_xticks(x)
            ax_impr.set_xticklabels(atms)
            ax_impr.set_title(f"{name} {sensor}: RMSE reduction vs ZOH [%]", fontsize=12, fontweight='bold')
            ax_impr.set_xlabel("Atmosphere")
            ax_impr.set_ylabel("RMSE reduction [%]")
            ax_impr.grid(True, linestyle=":", alpha=0.5, axis='y')
            ax_impr.legend(fontsize=8, loc='best')

        fig.suptitle(f"Slope prediction error, {sensor}", fontsize=15, fontweight='bold')
        self._save(fig, f"prediction_errors_comparison_{sensor}.png")
        return fig

    def _config_colors(self, configs):
        others = [c for c in configs if c != self.baseline]
        colors = {c: CONFIG_COLORS[i % len(CONFIG_COLORS)] for i, c in enumerate(others)}
        if self.baseline is not None:
            colors[self.baseline] = BASELINE_COLOR
        return colors

    def plot_closed_loop_strehl_contrast(self, df_cl_piv, configs, sensor):
        """
        Plots Closed-Loop Strehl Ratio (56 Hz) and Solar Contrast for every configuration, one sensor, noVibr.
        """
        df = df_cl_piv[(df_cl_piv['sensor'] == sensor) & (df_cl_piv['vibr'] == 'noVibr')]
        if df.empty:
            return None
        agg = self.aggregate_cl(df, ['atm'], configs)
        atms = sorted(agg['atm'], key=lambda a: int(a[3:]))
        agg = agg.set_index('atm').reindex(atms)
        present = [c for c in configs if f'Strehl_56Hz_{c}' in agg.columns]
        colors = self._config_colors(configs)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        x = np.arange(len(atms))
        width = 0.8 / max(len(present), 1)
        for ax, metric, title, ylabel in [(axes[0], 'Strehl_56Hz', f"Strehl ratio (56 Hz camera), {sensor}, noVibr", "Strehl ratio"),
                                          (axes[1], 'Contrast_Solar', f"Solar image contrast, {sensor}, noVibr", "Contrast")]:
            for k, c in enumerate(present):
                if f'{metric}_{c}' not in agg.columns:
                    continue
                ax.bar(x + (k - (len(present) - 1) / 2) * width, agg[f'{metric}_{c}'], width, label=c, color=colors[c], edgecolor='white', linewidth=0.5)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(atms)
            ax.set_xlabel("Atmosphere")
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle=":", alpha=0.5, axis='y')
            ax.legend(fontsize=9, loc='best')

        fig.suptitle(f"Closed-loop performance after {self.discard_time} s of transient, mean over draws", fontsize=14, fontweight='bold')
        self._save(fig, f"closed_loop_performance_comparison_{sensor}.png")
        return fig

    def plot_relative_gains_summary(self, df_cl_piv, configs, sensor):
        """
        Plots relative gains (%) in Strehl and Contrast of each configuration vs the baseline, one sensor, noVibr.
        """
        if self.baseline is None:
            return None
        df = df_cl_piv[(df_cl_piv['sensor'] == sensor) & (df_cl_piv['vibr'] == 'noVibr')]
        if df.empty:
            return None
        agg = self.aggregate_cl(df, ['atm'], configs)
        atms = sorted(agg['atm'], key=lambda a: int(a[3:]))
        agg = agg.set_index('atm').reindex(atms)
        others = [c for c in configs if c != self.baseline and f'Gain_SR_56Hz_{c}_pct' in agg.columns]
        if not others:
            return None
        colors = self._config_colors(configs)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        x = np.arange(len(atms))
        width = 0.8 / len(others)
        for ax, name, title in [(axes[0], 'Gain_SR_56Hz', "Strehl ratio (56 Hz)"), (axes[1], 'Gain_Contrast', "Solar contrast")]:
            for k, c in enumerate(others):
                if f'{name}_{c}_pct' not in agg.columns:
                    continue
                ax.bar(x + (k - (len(others) - 1) / 2) * width, agg[f'{name}_{c}_pct'], width, label=c, color=colors[c], edgecolor='white', linewidth=0.5)
            ax.axhline(0, color='black', linewidth=0.8)
            ax.set_title(f"{title}: relative change vs {self.baseline} [%]", fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(atms)
            ax.set_xlabel("Atmosphere")
            ax.set_ylabel("Relative change [%]")
            ax.grid(True, linestyle=":", alpha=0.5, axis='y')
            ax.legend(fontsize=9, loc='best')

        fig.suptitle(f"Gains relative to {self.baseline}, {sensor}, noVibr", fontsize=14, fontweight='bold')
        self._save(fig, f"closed_loop_relative_gains_{sensor}.png")
        return fig

    # ==============================================================================
    # 5. REPORT EXPORT (CSV, MARKDOWN & HTML)
    # ==============================================================================

    def export_reports(self, df_ol, df_sc, df_cl_raw, df_cl_piv, configs):
        """
        Exports all tabular metrics and comparative summaries to CSV, Markdown, and HTML reports.
        """
        reports = self.output_dir / "reports"
        for df, name in [(df_ol, "predictor_open_loop_metrics"), (df_sc, "predictor_pol_sin_cerrar_metrics"),
                         (df_cl_raw, "predictor_closed_loop_raw"), (df_cl_piv, "predictor_closed_loop_comparison")]:
            if not df.empty:
                df.to_csv(reports / f"{name}.csv", index=False)
        if self.excluded:
            pd.DataFrame(self.excluded, columns=['file', 'reason']).to_csv(reports / "excluded_runs.csv", index=False)

        tables = self._summary_tables(df_ol, df_sc, df_cl_piv, configs)
        with open(reports / "predictor_analysis_report.md", 'w', encoding='utf-8') as f:
            f.write(self._build_markdown_report(tables))
        with open(reports / "predictor_analysis_report.html", 'w', encoding='utf-8') as f:
            f.write(self._build_html_report(tables))

        self.logger.info("  [SAVED] Tabular CSVs, Markdown, and HTML Reports successfully.")

    def _summary_tables(self, df_ol, df_sc, df_cl_piv, configs):
        tables = {}
        err_cols = ['sensor', 'vibr', 'n_cases', 'RMSE_Baseline_ZOH', 'RMSE_Linear', 'Impr_RMSE_Linear_pct', 'RMSE_LSTM', 'Impr_RMSE_LSTM_pct',
                    'MAE_Baseline_ZOH', 'MAE_Linear', 'MAE_LSTM']
        tables['ol'] = self.aggregate_errors(df_ol, ['sensor', 'vibr'])[err_cols] if not df_ol.empty else pd.DataFrame()
        tables['sc'] = self.aggregate_errors(df_sc, ['sensor', 'vibr'])[err_cols] if not df_sc.empty else pd.DataFrame()
        if not df_cl_piv.empty:
            agg = self.aggregate_cl(df_cl_piv, ['sensor', 'vibr', 'atm'], configs)
            cols = ['sensor', 'vibr', 'atm']
            for c in configs:
                cols += [f'Strehl_56Hz_{c}', f'Gain_SR_56Hz_{c}_pct', f'Contrast_Solar_{c}', f'Gain_Contrast_{c}_pct', f'WFE_RMS_nm_{c}']
            tables['cl'] = agg[[c for c in cols if c in agg.columns]]
        else:
            tables['cl'] = pd.DataFrame()
        tables['excluded'] = pd.DataFrame(self.excluded, columns=['file', 'reason'])
        return tables

    @staticmethod
    def _format_value(col, val):
        if pd.isna(val):
            return "N/A"
        if col.endswith('_pct'):
            return f"{val:+.2f}%"
        if isinstance(val, (int, np.integer)):
            return str(val)
        if isinstance(val, (float, np.floating)):
            return f"{val:.4g}"
        return str(val)

    @classmethod
    def _df_to_md(cls, df):
        if df.empty:
            return "*No data*"
        headers = [str(col) for col in df.columns]
        lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(cls._format_value(col, row[col]) for col in df.columns) + " |")
        return "\n".join(lines)

    def _settings_lines(self):
        return [
            f"- Iteraciones esperadas por run: {self.n_iterations}. Los runs incompletos se excluyen (ver tabla final).",
            f"- Transitorio descartado: {self.discard_its} iteraciones ({self.discard_time} s); se descartan las exposiciones largas que lo solapan.",
            f"- Baseline de lazo cerrado: `{self.baseline}`.",
            "- Métrica de mejora única: reducción del RMSE frente a ZOH, calculada sobre RMSE agregados (RMS de los RMSE por caso).",
            "- 36x36 y 50x50 se reportan siempre por separado.",
        ]

    def _build_markdown_report(self, tables):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            f"# Informe de Rendimiento: Predictor de Pendientes AO ({self.base_dir.name})",
            f"\n*Generado automáticamente el {ts}*\n",
            "## 1. Configuración del análisis",
            *self._settings_lines(),
            "\n## 2. Error de predicción en lazo abierto (Open Loop)\n",
            self._df_to_md(tables['ol']),
            "\n## 3. Error de predicción en POL sin cerrar\n",
            self._df_to_md(tables['sc']),
            "\n## 4. Rendimiento en lazo cerrado (Strehl, contraste y WFE)\n",
            self._df_to_md(tables['cl']),
            "\n## 5. Runs excluidos\n",
            self._df_to_md(tables['excluded']),
            "\n---\n*Reporte generado por `analyze_predictor_performance.py`*",
        ]
        return "\n".join(lines)

    def _build_html_report(self, tables):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        def to_html(df):
            if df.empty:
                return "<p>No data</p>"
            formatted = df.copy().astype(object)
            for col in df.columns:
                formatted[col] = [self._format_value(col, v) for v in df[col]]
            return formatted.to_html(classes="styled-table", index=False)

        settings = "".join(f"<li>{line[2:]}</li>" for line in self._settings_lines())
        html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>AO Predictor Performance Report</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; background-color: #f8f9fa; color: #333; }}
  h1, h2, h3 {{ color: #1b3b6f; }}
  .header {{ background: linear-gradient(135deg, #1b3b6f, #4a90e2); color: white; padding: 25px; border-radius: 8px; margin-bottom: 30px; }}
  .card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 25px; overflow-x: auto; }}
  .styled-table {{ border-collapse: collapse; width: 100%; font-size: 0.9em; margin-top: 15px; }}
  .styled-table thead tr {{ background-color: #1b3b6f; color: white; text-align: left; }}
  .styled-table th, .styled-table td {{ padding: 10px 12px; border: 1px solid #ddd; }}
  .styled-table tbody tr:nth-of-type(even) {{ background-color: #f3f3f3; }}
  .styled-table tbody tr:hover {{ background-color: #e9f2ff; }}
</style>
</head>
<body>
<div class="header">
  <h1>AO Slope Predictor: Informe de Análisis y Rendimiento ({self.base_dir.name})</h1>
  <p>Generado: {ts} | Open Loop, POL sin cerrar y Closed Loop</p>
</div>

<div class="card">
  <h2>1. Configuración del análisis</h2>
  <ul>{settings}</ul>
</div>

<div class="card">
  <h2>2. Error de predicción en lazo abierto (Open Loop)</h2>
  {to_html(tables['ol'])}
</div>

<div class="card">
  <h2>3. Error de predicción en POL sin cerrar</h2>
  {to_html(tables['sc'])}
</div>

<div class="card">
  <h2>4. Rendimiento en lazo cerrado (Strehl, contraste y WFE)</h2>
  {to_html(tables['cl'])}
</div>

<div class="card">
  <h2>5. Runs excluidos</h2>
  {to_html(tables['excluded'])}
</div>
</body>
</html>
"""
        return html

    # ==============================================================================
    # PIPELINE EXECUTION
    # ==============================================================================

    def run_full_analysis(self):
        df_ol = self.analyze_open_loop()
        df_sc = self.analyze_cl_sin_cerrar()
        df_cl_raw, df_cl_piv = self.analyze_closed_loop()
        configs = sorted(df_cl_raw['config'].unique()) if not df_cl_raw.empty else []

        sensors = sorted(set().union(*[set(df['sensor']) for df in [df_ol, df_sc, df_cl_piv] if not df.empty]))
        for sensor in sensors:
            self.plot_prediction_errors_comparison(df_ol, df_sc, sensor)
            if not df_cl_piv.empty:
                self.plot_closed_loop_strehl_contrast(df_cl_piv, configs, sensor)
                self.plot_relative_gains_summary(df_cl_piv, configs, sensor)

        self.export_reports(df_ol, df_sc, df_cl_raw, df_cl_piv, configs)

        self.logger.info("=" * 80)
        self.logger.info(f" PREDICTOR ANALYSIS COMPLETED ({len(self.excluded)} runs excluded)")
        self.logger.info(f" Figures & Reports saved to: {self.output_dir}")
        self.logger.info("=" * 80)

        return df_ol, df_sc, df_cl_raw, df_cl_piv


# ==============================================================================
# CLI ENTRY POINT
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="AO Slope Predictor Performance Analysis")
    parser.add_argument('--base_dir', type=str, default=None, help="Results directory containing the cl_* folders (default: ~/simulations/results/thirdAttempt)")
    parser.add_argument('--ol_dir', type=str, default=None, help="Open-loop results directory (default: <base_dir>/predictor_ol or ~/simulations/results/predictor_ol)")
    parser.add_argument('--output_dir', type=str, default=None, help="Output directory for reports and figures (default: <base_dir>/analysis_results)")
    parser.add_argument('--baseline', type=str, default=None, help="Baseline folder for the relative gains (default: cl_baseline / cl_baseline_gain0.25)")
    parser.add_argument('--n_iterations', type=int, default=2500, help="Expected iterations per run, shorter runs are excluded (default 2500)")
    parser.add_argument('--sampling_freq', type=float, default=2000.0, help="Loop sampling frequency [Hz] (default 2000)")
    parser.add_argument('--discard_time', type=float, default=0.2, help="Initial loop transient excluded from the metrics [s] (default 0.2)")
    return parser.parse_args()


def main():
    args = parse_args()
    analyzer = PredictorPerformanceAnalyzer(
        base_dir=args.base_dir,
        ol_dir=args.ol_dir,
        output_dir=args.output_dir,
        baseline=args.baseline,
        n_iterations=args.n_iterations,
        sampling_freq=args.sampling_freq,
        discard_time=args.discard_time
    )
    analyzer.run_full_analysis()


if __name__ == '__main__':
    main()
