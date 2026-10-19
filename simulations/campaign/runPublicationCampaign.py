#!/usr/bin/env python3
"""
Publication campaign of the wavefront predictor (campaign v3: SAOS WFS with partially illuminated subapertures, develop 3534fa7).

Stages (run in this order by --stages all; every stage skips what already exists, so the job can be resubmitted):
  smoke    short runs of the new code paths (noise, 700 Hz, wind scale, open loop, dataset); the campaign stops if one fails
  assets   IMs of the 36x36 and 50x50 sensors for v3 (prepare_assets.py), phase screens of every atmosphere/wind used, and the 700 Hz
           DM models and vibrations (generated once here so that the parallel runs never write the same file)
  dataset  dataset v3 (generateTrainingData.py): 48 train + 8 validation runs, 36x36, random conditions per seed
  gains    gain scan of the baselines, per scenario and sensor, on 3 atmospheres (r0 5.5, 10.9, 22.9 cm; draw 1), 600 iterations:
           integrator gains INT_GAINS, POL without predictor (ZOH) gains POL_GAINS
  train    LSTM v2 with the band loss (tag v3) and with the MSE loss (tag v3_mse, ablation), increment AR with the band weighting
  choose   best gains (lowest mean WFE over the scan atmospheres) -> <results>/campaign/gains.json
  cl       paired closed loop (closedLoopComparison.py), every scenario x sensor x the 15 validation atmospheres (5 r0 x 3 wind
           profiles), 1000 iterations (200 discarded): integrator (best gain), POL with ZOH / linear / incAR / LSTM v3 / LSTM v3 mse,
           all POL cases with the best POL-ZOH gain of the scenario
  ol       open-loop prediction errors (polPredictionErrors.py: OL truth, POL input), every scenario x sensor x 5 atmospheres (draw 1)

Scenarios: S1 2 kHz no vibrations; S2 2 kHz vibrations; S3 2 kHz vibrations, wind x2; S4 700 Hz vibrations, wind x0.75 (the layers
move at most 1 px per step, 14.5 m/s at 700 Hz: x0.75 keeps every layer below it); S5 2 kHz no vibrations, noisy WFS (lightRatio
5e-4: ~0.03 px rms slope noise, 40-70 DN/px). Sensors 36x36 (ASM, trained on) and 50x50 (M7, never seen in training).

Usage (inside one Slurm job with 1 GPU and 48 CPUs, see jobs/job_publication_v3.job):
  python -u runPublicationCampaign.py --stages all --workers 12
  python -u runPublicationCampaign.py --stages cl --dry-run
"""
import os
import sys
import json
import glob
import time
import argparse
import subprocess
import concurrent.futures as cf
import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.dirname(HERE)
REPO = os.path.dirname(SIM)
sys.path.insert(0, SIM)
os.environ.setdefault('WFEPRED_CAMPAIGN', 'v3')

import common                                     # noqa: E402  (after WFEPRED_CAMPAIGN)
from atmosphereCases import atm_cases             # noqa: E402

SCENARIOS = {
    'S1': dict(vib=False, wind=1.0, fs=2000.0, noise=None),
    'S2': dict(vib=True, wind=1.0, fs=2000.0, noise=None),
    'S3': dict(vib=True, wind=2.0, fs=2000.0, noise=None),
    'S4': dict(vib=True, wind=0.75, fs=700.0, noise=None),
    'S5': dict(vib=False, wind=1.0, fs=2000.0, noise=5e-4),
}
SENSORS = (36, 50)
ATMS = [f'atm{i}' for i in range(1, 6)]
DRAWS = [f'draw{i}' for i in range(1, 4)]
SCAN_ATMS = ['atm1', 'atm3', 'atm5']
INT_GAINS = (0.15, 0.25, 0.35, 0.5)
POL_GAINS = (0.3, 0.5, 0.7, 1.0)
PREDICTOR_CASES = ['pol:zoh', 'pol:linear', 'pol:incAR:incAR_p16_lp100_hf0.1_v3', 'pol:lstm_v2:v3', 'pol:lstm_v2:v3_mse']
OL_PREDICTORS = ['zoh', 'linear', 'incAR:incAR_p16_lp100_hf0.1_v3', 'lstm_v2:v3', 'lstm_v2:v3_mse']
PY = sys.executable


def scenario_args(sc):
    a = []
    if sc['vib']:
        a.append('--vibrations')
    if sc['wind'] != 1:
        a += ['--wind_scale', f"{sc['wind']}"]
    if sc['fs'] != common.FS:
        a += ['--fs', f"{sc['fs']:g}"]
    if sc['noise'] is not None:
        a += ['--wfs_noise', f"{sc['noise']:g}"]
    return a


def run_tasks(tasks, workers, log_dir, threads, dry_run=False, abort_on_failure=False):
    """tasks: list of (name, cmd list). Runs them with `workers` processes at a time; each one logs to log_dir/name.log."""
    os.makedirs(log_dir, exist_ok=True)
    if dry_run:
        for name, cmd in tasks:
            print(f'[dry-run] {name}: {" ".join(cmd)}')
        return
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads),
               PYTHONPATH=os.pathsep.join([os.path.expanduser('~/code/SAOS'), REPO, os.environ.get('PYTHONPATH', '')]))

    def one(task):
        name, cmd = task
        t0 = time.time()
        with open(os.path.join(log_dir, name + '.log'), 'w') as log:
            rc = subprocess.run(cmd, cwd=SIM, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        return name, rc, time.time() - t0

    t0 = time.time()
    failed = []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for k, (name, rc, dt) in enumerate(ex.map(one, tasks), 1):
            print(f'  [{k}/{len(tasks)}] {name}: rc {rc}, {dt / 60:.1f} min (elapsed {(time.time() - t0) / 3600:.2f} h)', flush=True)
            if rc != 0:
                failed.append(name)
    if failed:
        print(f'  FAILED ({len(failed)}): {", ".join(failed)}', flush=True)
        if abort_on_failure:
            raise SystemExit(f'Aborted: {len(failed)} failed tasks, see the logs in {log_dir}')


def stage_assets(args, base):
    from SAOS.LoggingHelper import LoggingHelper
    import logging
    print('== assets: IMs v3', flush=True)
    if not args.dry_run:
        rc = subprocess.run([PY, '-u', 'prepare_assets.py', '--base_dir', base], cwd=SIM,
                            env=dict(os.environ, PYTHONPATH=os.pathsep.join([os.path.expanduser('~/code/SAOS'), REPO, os.environ.get('PYTHONPATH', '')]))).returncode
        if rc != 0:
            raise SystemExit('prepare_assets.py failed')
    lh = LoggingHelper(logging.WARNING)
    tel = common.make_telescope(lh.logger)
    mm, vd = common.get_asset_dirs(base)
    print('== assets: 700 Hz DM models and vibrations', flush=True)
    for fs in sorted({sc['fs'] for sc in SCENARIOS.values()} - {common.FS}):
        for m in ('asm', 'm7'):
            common.resampled_asset(os.path.join(mm, f'{m}_discrete_model.h5'), fs, 'dm')
        for i in range(1, 6):
            common.resampled_asset(os.path.join(vd, f'EST_vibration_{i}.h5'), fs, 'vibration')
    print('== assets: phase screens', flush=True)
    ps = os.path.join(base, 'phase_screens')
    jobs = []
    for ws in sorted({sc['wind'] for sc in SCENARIOS.values()}):
        for atm in ATMS:
            for draw in DRAWS:
                kw = dict(atm_cases[atm][draw])
                if ws != 1:
                    kw['windSpeed'] = [v * ws for v in kw['windSpeed']]
                # Same names and seeds as closedLoopComparison.atmosphere_choice and polPredictionErrors.run_case
                cl_seed = 100 * int(atm[3:]) + int(draw[4:])
                jobs.append((os.path.join(ps, f'ps_wfePredictor_mini_{atm}_{draw}' + (f'_wind{ws}' if ws != 1 else '') + '.h5'), kw, cl_seed))
                if draw == 'draw1':
                    ol_seed = 2026 + 10 * int(atm[3:]) + int(draw[4:])
                    jobs.append((os.path.join(ps, f'ps_polpred_{atm}_{draw}_seed{ol_seed}' + (f'_wind{ws:g}' if ws != 1 else '') + '.h5'), kw, ol_seed))
    todo = [j for j in jobs if not os.path.exists(j[0])]
    print(f'   {len(jobs)} phase-screen files, {len(todo)} to generate', flush=True)
    if not args.dry_run:
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(lambda j: common.load_atmosphere(tel, j[1], j[0], j[2], lh.logger), todo))
    lh.queue_listerner.stop()


def smoke_tasks(base):
    """Short runs of every new code path (noise, 700 Hz, wind scale, open loop) before the long stages."""
    t = []
    for sname in ('S3', 'S4', 'S5'):
        t.append((f'smoke_cl_{sname}', [PY, '-u', 'closedLoopComparison.py', '--test', '--sensor', '36', '--cases', 'integrator', 'pol:zoh', 'pol:linear',
                                        '--base_dir', base] + scenario_args(SCENARIOS[sname])))
    t.append(('smoke_cl_S2_50', [PY, '-u', 'closedLoopComparison.py', '--test', '--sensor', '50', '--cases', 'pol:zoh', '--base_dir', base]
              + scenario_args(SCENARIOS['S2'])))
    t.append(('smoke_ol_S4', [PY, '-u', 'polPredictionErrors.py', '--sensor', '36', '--atm', 'atm1', '--draw', 'draw1', '--predictors', 'zoh', 'linear',
                              '--n_iterations', '100', '--discard', '0.02', '--vibr_only', '--wind_scale', '0.75', '--fs', '700', '--base_dir', base,
                              '--results_subdir', f'wfePredictor/{common.CAMPAIGN}/smoke']))
    t.append(('smoke_ol_S5', [PY, '-u', 'polPredictionErrors.py', '--sensor', '50', '--atm', 'atm1', '--draw', 'draw1', '--predictors', 'zoh', 'linear',
                              '--n_iterations', '100', '--discard', '0.02', '--no_vibr_only', '--wfs_noise', '0.0005', '--base_dir', base,
                              '--results_subdir', f'wfePredictor/{common.CAMPAIGN}/smoke']))
    t.append(('smoke_dataset', [PY, '-u', 'generateTrainingData.py', '--test', '--split', 'train', '--sensor', '36', '--index', '0', '--base_dir', base]))
    return t


def dataset_tasks(base):
    return [(f'dataset_{split}_{i}', [PY, '-u', 'generateTrainingData.py', '--split', split, '--sensor', '36', '--index', str(i),
                                      '--base_dir', base, '--skip_existing'])
            for split, n in (('train', 48), ('val', 8)) for i in range(n)]


def gain_tasks(base, int_gains=INT_GAINS, pol_gains=POL_GAINS, combos=None):
    tasks = []
    for sname, sc in SCENARIOS.items():
        for s in SENSORS:
            if combos and f'{sname}_{s}' not in combos:
                continue
            for atm in SCAN_ATMS:
                common_args = ['--sensor', str(s), '--atm', atm, '--draw', 'draw1', '--n_iterations', '600', '--discard', '200',
                               '--base_dir', base, '--skip_existing'] + scenario_args(sc)
                for g in int_gains:
                    tasks.append((f'gains_{sname}_{s}_{atm}_int{g}', [PY, '-u', 'closedLoopComparison.py', '--cases', 'integrator',
                                                                    '--int_gain', str(g)] + common_args))
                for g in pol_gains:
                    tasks.append((f'gains_{sname}_{s}_{atm}_pol{g}', [PY, '-u', 'closedLoopComparison.py', '--cases', 'pol:zoh',
                                                                    '--gain', str(g)] + common_args))
    return tasks


def train_tasks(base):
    data = os.path.join(common.results_root(base), 'dataset')
    w = os.path.join(REPO, 'wfePredictorSAOS', 'weights')
    tasks = []
    if not os.path.exists(os.path.join(w, 'lstm_v2_v3.pt')):
        tasks.append(('train_lstm_band', [PY, '-u', os.path.join(REPO, 'wfePredictorSAOS', 'training', 'trainIncrementalLSTM.py'),
                                          '--data_dir', data, '--loss', 'band', '--tag', 'v3']))
    if not os.path.exists(os.path.join(w, 'lstm_v2_v3_mse.pt')):
        tasks.append(('train_lstm_mse', [PY, '-u', os.path.join(REPO, 'wfePredictorSAOS', 'training', 'trainIncrementalLSTM.py'),
                                         '--data_dir', data, '--loss', 'mse', '--tag', 'v3_mse']))
    if not os.path.exists(os.path.join(w, 'incAR_p16_lp100_hf0.1_v3.npz')):
        tasks.append(('fit_incAR_band', [PY, '-u', os.path.join(REPO, 'wfePredictorSAOS', 'predictor', 'incrementARPredictor.py'),
                                         '--files', os.path.join(data, 'train', '*.h5'), '--p', '16', '--lowpass', '100', '--hf_weight', '0.1',
                                         '--out', os.path.join(w, 'incAR_p16_lp100_hf0.1_v3.npz')]))
    return tasks


def scan_wfe(base, sname, s, case, gain_key, g):
    """Mean WFE over the scan atmospheres for one baseline gain (None if any run is missing or diverged)."""
    vals = []
    for atm in SCAN_ATMS:
        pat = os.path.join(common.results_root(base), 'closedLoopComparison', f'{s}x{s}_{atm}_draw1*_600it_*{gain_key}{g}*', f'res_{case}.h5')
        hits = []
        for p in glob.glob(pat):
            with h5py.File(p, 'r') as f:
                cfg = json.loads(f.attrs['config']); sm = json.loads(f.attrs['summary'])
            sc = SCENARIOS[sname]
            if bool(cfg.get('vibrations')) == sc['vib'] and float(cfg.get('wind_scale', 1)) == sc['wind'] and float(cfg.get('fs', 2000)) == sc['fs'] \
                    and cfg.get('wfs_noise') == sc['noise'] and float(cfg['case_gain']) == g:
                hits.append(np.inf if sm['diverged'] else sm['wfe_nm'])
        if not hits:
            return None
        vals.append(min(hits))
    return float(np.mean(vals))


def stage_choose(base, int_gains=INT_GAINS, pol_gains=POL_GAINS):
    out = {}
    for sname in SCENARIOS:
        for s in SENSORS:
            row = {}
            for case, key, grid in (('integrator', 'intg', int_gains), ('pol_zoh', 'polg', pol_gains)):
                w = {g: scan_wfe(base, sname, s, case, key, g) for g in grid}
                ok = {g: v for g, v in w.items() if v is not None and np.isfinite(v)}
                row[case] = dict(best=min(ok, key=ok.get) if ok else None, wfe_nm={str(g): v for g, v in w.items()})
            out[f'{sname}_{s}'] = row
            print(f"  {sname} {s}x{s}: integrator best g {row['integrator']['best']} | POL best g {row['pol_zoh']['best']}", flush=True)
    path = os.path.join(common.results_root(base), 'campaign', 'gains.json')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):                       # keep the previous choice
        os.replace(path, path.replace('.json', time.strftime('_%Y%m%d_%H%M%S.json')))
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    print(f'  saved {path}', flush=True)
    return out


def load_gains(base):
    with open(os.path.join(common.results_root(base), 'campaign', 'gains.json')) as f:
        return json.load(f)


def cl_tasks(base, gains, combos=None, cases=None, pol_gain=None):
    """pol_gain: POL gain of every run instead of the chosen one (e.g. a lower, stable gain for the predictors); own task/log names."""
    tasks = []
    cases = cases or ['integrator', *PREDICTOR_CASES]
    for sname, sc in SCENARIOS.items():
        for s in SENSORS:
            if combos and f'{sname}_{s}' not in combos:
                continue
            g = gains[f'{sname}_{s}']
            for atm in ATMS:
                for draw in DRAWS:
                    gp = g['pol_zoh']['best'] if pol_gain is None else pol_gain
                    tasks.append((f'cl_{sname}_{s}_{atm}_{draw}' + (f'_polg{pol_gain:g}' if pol_gain is not None else ''),
                                  [PY, '-u', 'closedLoopComparison.py', '--sensor', str(s), '--atm', atm, '--draw', draw,
                                  '--cases', *cases, '--int_gain', str(g['integrator']['best']), '--gain', str(gp),
                                  '--n_iterations', '1000', '--discard', '200', '--base_dir', base, '--skip_existing'] + scenario_args(sc)))
    return tasks


def ol_tasks(base, gains):
    tasks = []
    for sname, sc in SCENARIOS.items():
        for s in SENSORS:
            g = gains[f'{sname}_{s}']['pol_zoh']['best']
            extra = ['--vibr_only' if sc['vib'] else '--no_vibr_only']
            if sc['wind'] != 1:
                extra += ['--wind_scale', f"{sc['wind']}"]
            if sc['fs'] != common.FS:
                extra += ['--fs', f"{sc['fs']:g}"]
            if sc['noise'] is not None:
                extra += ['--wfs_noise', f"{sc['noise']:g}"]
            for atm in ATMS:
                tasks.append((f'ol_{sname}_{s}_{atm}', [PY, '-u', 'polPredictionErrors.py', '--sensor', str(s), '--atm', atm, '--draw', 'draw1',
                              '--predictors', *OL_PREDICTORS, '--gain', str(g), '--n_iterations', '1000', '--discard', '0.1',
                              '--base_dir', base, '--skip_existing'] + extra))
    return tasks


def main():
    parser = argparse.ArgumentParser(description='Publication campaign of the predictor (v3)')
    parser.add_argument('--stages', nargs='+', default=['all'], choices=['all', 'assets', 'smoke', 'dataset', 'gains', 'train', 'choose', 'cl', 'ol'])
    parser.add_argument('--workers', type=int, default=12, help='Simulations at the same time on the GPU (default 12)')
    parser.add_argument('--threads', type=int, default=4, help='CPU threads per simulation (default 4)')
    parser.add_argument('--base_dir', default=None)
    parser.add_argument('--int-gains', nargs='+', type=float, default=list(INT_GAINS), help='Integrator gains of the scan / choice')
    parser.add_argument('--pol-gains', nargs='+', type=float, default=list(POL_GAINS), help='POL (ZOH) gains of the scan / choice')
    parser.add_argument('--combos', nargs='+', default=None, help='Only these scenario_sensor combinations in gains and cl, e.g. S2_50 S4_36')
    parser.add_argument('--cl-cases', nargs='+', default=None, help='Closed-loop cases to run (default: integrator and every POL case)')
    parser.add_argument('--cl-pol-gain', type=float, default=None, help='POL gain of the closed-loop runs instead of gains.json (with --combos)')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    base = common.get_base_dir(args.base_dir)
    stages = ['assets', 'smoke', 'dataset', 'gains', 'train', 'choose', 'cl', 'ol'] if 'all' in args.stages else args.stages
    logs = os.path.join(base, 'results', 'logs', 'wfePredictor', common.CAMPAIGN)
    print(f'Campaign {common.CAMPAIGN} | base {base} | results {common.results_root(base)} | logs {logs} | stages {stages}', flush=True)
    t0 = time.time()
    if 'assets' in stages:
        stage_assets(args, base)
    if 'smoke' in stages:
        t = smoke_tasks(base)
        print(f'== smoke tests: {len(t)} short runs', flush=True)
        run_tasks(t, args.workers, logs, args.threads, args.dry_run, abort_on_failure=True)
    # The dataset and the gain scan are independent: one pool for both
    pool = (dataset_tasks(base) if 'dataset' in stages else []) + (gain_tasks(base, args.int_gains, args.pol_gains, args.combos) if 'gains' in stages else [])
    if pool:
        print(f'== dataset + gains: {len(pool)} runs', flush=True)
        run_tasks(pool, args.workers, logs, args.threads, args.dry_run)
    if 'train' in stages:
        t = train_tasks(base)
        print(f'== train: {len(t)} tasks', flush=True)
        run_tasks(t, 3, logs, max(args.threads, 8), args.dry_run)
    gains = None
    if 'choose' in stages and not args.dry_run:
        print('== choose gains', flush=True)
        gains = stage_choose(base, args.int_gains, args.pol_gains)
    if ('cl' in stages or 'ol' in stages) and gains is None:
        gains = load_gains(base) if not args.dry_run else {f'{a}_{s}': {'integrator': {'best': 0.25}, 'pol_zoh': {'best': 0.4}} for a in SCENARIOS for s in SENSORS}
    if 'cl' in stages:
        t = cl_tasks(base, gains, args.combos, args.cl_cases, args.cl_pol_gain)
        print(f'== closed loop: {len(t)} runs', flush=True)
        run_tasks(t, args.workers, logs, args.threads, args.dry_run)
    if 'ol' in stages:
        t = ol_tasks(base, gains)
        print(f'== open loop: {len(t)} runs', flush=True)
        run_tasks(t, args.workers, logs, args.threads, args.dry_run)
    print(f'Done in {(time.time() - t0) / 3600:.2f} h', flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
