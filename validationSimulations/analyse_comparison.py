#!/usr/bin/env python3
import os
import sys
import glob
import json
import h5py
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Unified Analysis for 2kHz AO Predictor Simulations")
    parser.add_argument('--sensor', type=str, default=None, choices=['36', '50', 'all'], help="Filter by sensor (36, 50, or all)")
    parser.add_argument('--base_dir', type=str, default=None, help="Base directory for results (default: /mnt/nas-mcao/predictor_sims/results or ~/simulations/results)")
    return parser.parse_args()

def get_results_base_dir(custom_base_dir=None):
    if custom_base_dir:
        if custom_base_dir.endswith('results'):
            return custom_base_dir
        return os.path.join(custom_base_dir, 'results')
    if os.path.exists('/mnt/nas-mcao'):
        return '/mnt/nas-mcao/predictor_sims/results'
    return os.path.expanduser('~/simulations/results')

def analyze_open_loop(base_dir, sensor_filter=None):
    ol_dir = os.path.join(base_dir, 'predictor_ol')
    if not os.path.exists(ol_dir):
        return []

    metric_files = glob.glob(os.path.join(ol_dir, 'metrics_*.json'))
    results = []
    for mf in metric_files:
        try:
            with open(mf, 'r') as f:
                data = json.load(f)
            if sensor_filter and sensor_filter != 'all' and sensor_filter not in data.get('sensor', ''):
                continue
            results.append(data)
        except Exception as e:
            print(f"Error reading {mf}: {e}")

    results.sort(key=lambda x: (x.get('sensor', ''), x.get('vibr', ''), int(x.get('atm', 'atm0').replace('atm', '')), int(x.get('draw', 'draw0').replace('draw', ''))))
    return results

def analyze_cl_strehl(base_dir, folder_name, sensor_filter=None):
    folder_path = os.path.join(base_dir, folder_name)
    if not os.path.exists(folder_path):
        return []

    h5_files = glob.glob(os.path.join(folder_path, '*.h5'))
    results = []
    for fp in h5_files:
        fn = os.path.basename(fp).replace('.h5', '')
        parts = fn.split('_')
        
        sensor = "unknown"
        vibr = "unknown"
        atm = "unknown"
        draw = "unknown"

        for p in parts:
            if 'x' in p and (p.startswith('36') or p.startswith('50')):
                sensor = p
            elif p in ['noVibr', 'Vibr']:
                vibr = p
            elif p.startswith('atm'):
                atm = p
            elif p.startswith('draw'):
                draw = p

        if sensor_filter and sensor_filter != 'all' and sensor_filter not in sensor:
            continue

        try:
            with h5py.File(fp, 'r') as f:
                # 56 Hz camera (LightPath_1)
                strehl_56hz = None
                std_56hz = None
                p_56_long = 'LightPath_1/sci_frame_longExp/strehl'
                p_56_short = 'LightPath_1/sci_frame_shortExp/strehl'
                if p_56_long in f and len(f[p_56_long][()]) >= 1:
                    arr = f[p_56_long][()]
                    eval_arr = arr[1:] if len(arr) > 2 else arr
                    strehl_56hz = float(np.mean(eval_arr))
                    std_56hz = float(np.std(eval_arr))
                elif p_56_short in f and len(f[p_56_short][()]) > 2:
                    arr = f[p_56_short][()]
                    strehl_56hz = float(np.mean(arr[2:]))
                    std_56hz = float(np.std(arr[2:]))

                # 5 Hz ultra long exposure camera (LightPath_2)
                strehl_5hz = None
                std_5hz = None
                p_5_long = 'LightPath_2/sci_frame_longExp/strehl'
                if p_5_long in f and len(f[p_5_long][()]) >= 1:
                    arr = f[p_5_long][()]
                    eval_arr = arr[1:] if len(arr) > 2 else arr
                    strehl_5hz = float(np.mean(eval_arr))
                    std_5hz = float(np.std(eval_arr))

                if strehl_56hz is None and strehl_5hz is None:
                    continue

                results.append({
                    'case': folder_name,
                    'sensor': sensor,
                    'atm': atm,
                    'draw': draw,
                    'vibr': vibr,
                    'strehl_56hz': strehl_56hz,
                    'std_56hz': std_56hz,
                    'strehl_5hz': strehl_5hz,
                    'std_5hz': std_5hz
                })
        except Exception as e:
            print(f"Error reading {fp}: {e}")

    results.sort(key=lambda x: (x['sensor'], x['vibr'], int(x['atm'].replace('atm', '0') if 'atm' in x['atm'] else 0), int(x['draw'].replace('draw', '0') if 'draw' in x['draw'] else 0)))
    return results

def analyze_cl_sin_cerrar(base_dir, sensor_filter=None):
    sc_dir = os.path.join(base_dir, 'cl_sin_cerrar')
    if not os.path.exists(sc_dir):
        return []

    metric_files = glob.glob(os.path.join(sc_dir, 'metrics_*.json'))
    results = []
    for mf in metric_files:
        try:
            with open(mf, 'r') as f:
                data = json.load(f)
            if sensor_filter and sensor_filter != 'all' and sensor_filter not in data.get('sensor', ''):
                continue
            results.append(data)
        except Exception as e:
            print(f"Error reading {mf}: {e}")

    results.sort(key=lambda x: (x.get('sensor', ''), x.get('vibr', ''), int(x.get('atm', 'atm0').replace('atm', '')), int(x.get('draw', 'draw0').replace('draw', ''))))
    return results

def main():
    args = parse_args()
    base_dir = get_results_base_dir(args.base_dir)

    print("=" * 125)
    print("                      UNIFIED AO PREDICTOR SIMULATION ANALYSIS REPORT (2 kHz)")
    print("=" * 125)

    # 1. Open Loop Analysis
    ol_res = analyze_open_loop(base_dir, args.sensor)
    if ol_res:
        print("\n" + "#" * 125)
        print(" [1] OPEN LOOP (OL) PREDICTION ERRORS (Slope RMSE in px & % Improvement vs ZOH delay=2)")
        print("#" * 125)
        print(f"{'Sensor':<8} | {'Atm':<6} {'Draw':<6} {'Vibr':<8} | {'ZOH RMSE':<12} | {'Linear RMSE':<14} {'(Impr %)':<10} | {'LSTM RMSE':<14} {'(Impr %)':<10}")
        print("-" * 125)
        for r in ol_res:
            print(f"{r.get('sensor', ''):<8} | {r.get('atm', ''):<6} {r.get('draw', ''):<6} {r.get('vibr', ''):<8} | "
                  f"{r.get('rmse_zoh', 0):<12.5f} | "
                  f"{r.get('rmse_linear', 0):<14.5f} {r.get('impr_linear_pct', 0):>+8.2f}% | "
                  f"{r.get('rmse_lstm', 0):<14.5f} {r.get('impr_lstm_pct', 0):>+8.2f}%")

    # 2. Closed Loop Sin Cerrar Analysis
    sc_res = analyze_cl_sin_cerrar(base_dir, args.sensor)
    if sc_res:
        print("\n" + "#" * 125)
        print(" [2] CLOSED LOOP 'SIN CERRAR' (POL Slope RMSE in px & % Improvement vs ZOH)")
        print("#" * 125)
        print(f"{'Sensor':<8} | {'Atm':<6} {'Draw':<6} {'Vibr':<8} | {'POL ZOH RMSE':<14} | {'POL Lin RMSE':<14} {'(Impr %)':<10} | {'POL LSTM RMSE':<14} {'(Impr %)':<10}")
        print("-" * 125)
        for r in sc_res:
            print(f"{r.get('sensor', ''):<8} | {r.get('atm', ''):<6} {r.get('draw', ''):<6} {r.get('vibr', ''):<8} | "
                  f"{r.get('rmse_pol_zoh', 0):<14.5f} | "
                  f"{r.get('rmse_pol_linear', 0):<14.5f} {r.get('impr_linear_pct', 0):>+8.2f}% | "
                  f"{r.get('rmse_pol_lstm', 0):<14.5f} {r.get('impr_lstm_pct', 0):>+8.2f}%")

    # 3. Closed Loop Strehl Comparison (56 Hz and 5 Hz)
    cl_baseline = analyze_cl_strehl(base_dir, 'cl_baseline', args.sensor)
    cl_pol_lstm = analyze_cl_strehl(base_dir, 'cl_pol_lstm', args.sensor)
    cl_pol_lin = analyze_cl_strehl(base_dir, 'cl_pol_linear', args.sensor)

    all_cl = cl_baseline + cl_pol_lstm + cl_pol_lin
    if all_cl:
        print("\n" + "#" * 125)
        print(" [3] CLOSED LOOP SCIENCE STREHL RATIO COMPARISON (56 Hz & 5 Hz Long Exposures)")
        print("#" * 125)
        print(f"{'Configuration':<20} | {'Sensor':<8} | {'Atm':<6} {'Draw':<6} {'Vibr':<8} | {'Strehl @ 56 Hz (T=17.9ms)':<28} | {'Strehl @ 5 Hz (T=200ms)':<28}")
        print("-" * 125)
        for r in sorted(all_cl, key=lambda x: (x['sensor'], x['vibr'], x['atm'], x['draw'], x['case'])):
            s56_str = f"{r['strehl_56hz']:.5f} ± {r['std_56hz']:.5f}" if r['strehl_56hz'] is not None else "N/A"
            s5_str = f"{r['strehl_5hz']:.5f} ± {r['std_5hz']:.5f}" if r['strehl_5hz'] is not None else "N/A"
            print(f"{r['case']:<20} | {r['sensor']:<8} | {r['atm']:<6} {r['draw']:<6} {r['vibr']:<8} | {s56_str:<28} | {s5_str:<28}")

        print("\n" + "-" * 100)
        print(" GLOBAL AVERAGE STREHL SUMMARY BY CONFIGURATION")
        print("-" * 100)
        for folder in ['cl_baseline', 'cl_pol_linear', 'cl_pol_lstm']:
            for sens in (['36x36', '50x50'] if not args.sensor or args.sensor == 'all' else [f"{args.sensor}x{args.sensor}"]):
                subset_56 = [r['strehl_56hz'] for r in all_cl if r['case'] == folder and r['sensor'] == sens and r['strehl_56hz'] is not None]
                subset_5 = [r['strehl_5hz'] for r in all_cl if r['case'] == folder and r['sensor'] == sens and r['strehl_5hz'] is not None]
                s56_summary = f"{np.mean(subset_56):.5f} ± {np.std(subset_56):.5f}" if subset_56 else "N/A"
                s5_summary = f"{np.mean(subset_5):.5f} ± {np.std(subset_5):.5f}" if subset_5 else "N/A"
                if subset_56 or subset_5:
                    print(f" -> {folder:<18} [{sens}]: Strehl@56Hz = {s56_summary:<20} | Strehl@5Hz = {s5_summary}")

    print("=" * 125)

if __name__ == "__main__":
    main()
