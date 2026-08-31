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
    return parser.parse_args()

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
        # Formats:
        # res_cl_baseline_2delay_36x36_noVibr_atm1_draw1
        # res_cl_pol_lstm_36x36_noVibr_atm1_draw1
        # res_cl_sin_cerrar_36x36_noVibr_atm1_draw1
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
                path_strehl_long = 'LightPath_1/sci_frame_longExp/strehl'
                path_strehl_short = 'LightPath_1/sci_frame_shortExp/strehl'

                if path_strehl_long in f and len(f[path_strehl_long][()]) >= 1:
                    arr = f[path_strehl_long][()]
                    # Use all long exp frames if available, or discard 1st transient
                    eval_arr = arr[1:] if len(arr) > 2 else arr
                    avg_s = float(np.mean(eval_arr))
                    std_s = float(np.std(eval_arr))
                elif path_strehl_short in f and len(f[path_strehl_short][()]) > 2:
                    arr = f[path_strehl_short][()]
                    avg_s = float(np.mean(arr[2:]))
                    std_s = float(np.std(arr[2:]))
                else:
                    continue

                results.append({
                    'case': folder_name,
                    'sensor': sensor,
                    'atm': atm,
                    'draw': draw,
                    'vibr': vibr,
                    'strehl': avg_s,
                    'std': std_s
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
    base_dir = os.path.expanduser('~/simulations/results')

    print("=" * 115)
    print("           UNIFIED AO PREDICTOR SIMULATION ANALYSIS REPORT (2 kHz)")
    print("=" * 115)

    # 1. Open Loop Analysis
    ol_res = analyze_open_loop(base_dir, args.sensor)
    if ol_res:
        print("\n" + "#" * 115)
        print(" [1] OPEN LOOP (OL) PREDICTION ERRORS (Slope RMSE in px & % Improvement vs ZOH delay=2)")
        print("#" * 115)
        print(f"{'Sensor':<8} | {'Atm':<6} {'Draw':<6} {'Vibr':<8} | {'ZOH RMSE':<12} | {'Linear RMSE':<14} {'(Impr %)':<10} | {'LSTM RMSE':<14} {'(Impr %)':<10}")
        print("-" * 115)
        for r in ol_res:
            print(f"{r.get('sensor', ''):<8} | {r.get('atm', ''):<6} {r.get('draw', ''):<6} {r.get('vibr', ''):<8} | "
                  f"{r.get('rmse_zoh', 0):<12.5f} | "
                  f"{r.get('rmse_linear', 0):<14.5f} {r.get('impr_linear_pct', 0):>+8.2f}% | "
                  f"{r.get('rmse_lstm', 0):<14.5f} {r.get('impr_lstm_pct', 0):>+8.2f}%")

    # 2. Closed Loop Sin Cerrar Analysis
    sc_res = analyze_cl_sin_cerrar(base_dir, args.sensor)
    if sc_res:
        print("\n" + "#" * 115)
        print(" [2] CLOSED LOOP 'SIN CERRAR' (POL Slope RMSE in px & % Improvement vs ZOH)")
        print("#" * 115)
        print(f"{'Sensor':<8} | {'Atm':<6} {'Draw':<6} {'Vibr':<8} | {'POL ZOH RMSE':<14} | {'POL Lin RMSE':<14} {'(Impr %)':<10} | {'POL LSTM RMSE':<14} {'(Impr %)':<10}")
        print("-" * 115)
        for r in sc_res:
            print(f"{r.get('sensor', ''):<8} | {r.get('atm', ''):<6} {r.get('draw', ''):<6} {r.get('vibr', ''):<8} | "
                  f"{r.get('rmse_pol_zoh', 0):<14.5f} | "
                  f"{r.get('rmse_pol_linear', 0):<14.5f} {r.get('impr_linear_pct', 0):>+8.2f}% | "
                  f"{r.get('rmse_pol_lstm', 0):<14.5f} {r.get('impr_lstm_pct', 0):>+8.2f}%")

    # 3. Closed Loop Strehl Comparison
    cl_baseline = analyze_cl_strehl(base_dir, 'cl_baseline', args.sensor)
    cl_pol_lstm = analyze_cl_strehl(base_dir, 'cl_pol_lstm', args.sensor)
    cl_pol_lin = analyze_cl_strehl(base_dir, 'cl_pol_linear', args.sensor)

    all_cl = cl_baseline + cl_pol_lstm + cl_pol_lin
    if all_cl:
        print("\n" + "#" * 115)
        print(" [3] CLOSED LOOP SCIENCE STREHL RATIO COMPARISON")
        print("#" * 115)
        print(f"{'Configuration':<25} | {'Sensor':<8} | {'Atm':<6} {'Draw':<6} {'Vibr':<8} | {'Strehl (Mean ± Std)':<25}")
        print("-" * 115)
        for r in sorted(all_cl, key=lambda x: (x['sensor'], x['vibr'], x['atm'], x['draw'], x['case'])):
            strehl_str = f"{r['strehl']:.5f} ± {r['std']:.5f}"
            print(f"{r['case']:<25} | {r['sensor']:<8} | {r['atm']:<6} {r['draw']:<6} {r['vibr']:<8} | {strehl_str:<25}")

        print("\n" + "-" * 80)
        print(" GLOBAL AVERAGE STREHL SUMMARY BY CONFIGURATION")
        print("-" * 80)
        for folder in ['cl_baseline', 'cl_pol_linear', 'cl_pol_lstm']:
            for sens in (['36x36', '50x50'] if not args.sensor or args.sensor == 'all' else [f"{args.sensor}x{args.sensor}"]):
                subset = [r['strehl'] for r in all_cl if r['case'] == folder and r['sensor'] == sens]
                if subset:
                    print(f" -> {folder:<20} [{sens}]: Average Strehl = {np.mean(subset):.5f} ± {np.std(subset):.5f} (N={len(subset)})")

    print("\n" + "=" * 115)

if __name__ == '__main__':
    main()
