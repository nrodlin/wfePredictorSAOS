#!/usr/bin/env python3
"""
Master Campaign Runner for AO Predictor Simulations (Durum / Local)
- Automatically ensures all assets (atmospheres, modal bases, IMs) are pre-generated
- Iterates over all requested sensors, atmospheric cases, draws, and vibrations
- Runs:
    1. Open Loop (OL)
    2. Closed Loop Baseline (ZOH delay=2)
    3. Closed Loop POL (Linear Predictor)
    4. Closed Loop POL (LSTM Predictor)
    5. Closed Loop Sin Cerrar (Parallel Open Predictor Monitoring)
- Concludes with the full summary analysis report via analyse_comparison.py
- Uses structured logging (LoggingHelper)
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import sys
import time
import argparse
import logging
import subprocess

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from SAOS.LoggingHelper import LoggingHelper
try:
    from prepare_assets import prepare_all_assets, get_base_dir
except ImportError:
    from validationSimulations.prepare_assets import prepare_all_assets, get_base_dir

def parse_args():
    parser = argparse.ArgumentParser(description="Master Campaign Runner for 2kHz AO Predictor Simulations")
    parser.add_argument('--base_dir', type=str, default=None, help="Base directory (default: /mnt/nas-mcao/predictor_sims or ~/simulations)")
    parser.add_argument('--sensors', nargs='+', type=int, default=[36, 50], help="Sensor sizes to run (default: 36 50)")
    parser.add_argument('--n_iterations', type=int, default=2500, help="Number of simulation iterations (default: 2500 for 1.25s)")
    parser.add_argument('--sampling_freq', type=float, default=2000.0, help="Sampling frequency in Hz (default: 2000.0)")
    parser.add_argument('--atm', type=str, default=None, help="Run specific atmosphere (e.g. atm1). Default: all")
    parser.add_argument('--draw', type=str, default=None, help="Run specific draw (e.g. draw1). Default: all")
    parser.add_argument('--skip_existing', action='store_true', help="Skip simulations if result file already exists")
    parser.add_argument('--force_assets', action='store_true', help="Force regenerating all assets at startup")
    parser.add_argument('--test', action='store_true', help="Run in test mode (50 iterations, atm1 draw1 noVibr)")
    return parser.parse_args()

def run_step(cmd_args, logger, step_name):
    logger.info(f"\n>>> Starting {step_name}...")
    logger.info(f"Command: {' '.join(cmd_args)}")
    t0 = time.time()
    result = subprocess.run(cmd_args, stdout=sys.stdout, stderr=sys.stderr)
    elapsed = time.time() - t0
    if result.returncode != 0:
        logger.error(f"FAILED {step_name} with exit code {result.returncode} in {elapsed:.2f} s")
        return False
    logger.info(f">>> COMPLETED {step_name} in {elapsed:.2f} s ({elapsed/60.0:.2f} min)")
    return True

def main():
    args = parse_args()

    test_logger = LoggingHelper(logging.INFO)
    logger = test_logger.logger

    base_dir = get_base_dir(args.base_dir)
    python_exec = sys.executable
    script_dir = os.path.dirname(os.path.abspath(__file__))

    logger.info("=" * 100)
    logger.info(" STARTING UNIFIED AO PREDICTOR SIMULATION CAMPAIGN")
    logger.info(f" Host: {os.uname().nodename}")
    logger.info(f" Base Directory: {base_dir}")
    logger.info(f" Sensors: {args.sensors}")
    logger.info(f" Sampling Frequency: {args.sampling_freq} Hz")
    logger.info(f" Iterations per simulation: {50 if args.test else args.n_iterations}")
    logger.info(f" Skip existing: {args.skip_existing}")
    logger.info(f" Test mode: {args.test}")
    logger.info("=" * 100)

    # -------------------------------------------------------------------------
    # STEP 1: PREPARE ALL ASSETS (Screens, Modal Bases, IMs)
    # -------------------------------------------------------------------------
    logger.info("\n>>> [STEP 1/3] Verifying and Pre-generating All Required Assets...")
    prepare_all_assets(
        base_dir=base_dir,
        force_atm=args.force_assets,
        force_modal=args.force_assets,
        force_im=args.force_assets,
        logger=logger
    )

    # -------------------------------------------------------------------------
    # STEP 2: EXECUTE SIMULATIONS ACROSS SENSORS
    # -------------------------------------------------------------------------
    logger.info("\n>>> [STEP 2/3] Launching Simulation Sequences...")
    campaign_start_time = time.time()

    common_flags = [
        '--base_dir', base_dir,
        '--n_iterations', str(args.n_iterations),
        '--sampling_freq', str(args.sampling_freq)
    ]
    if args.atm:
        common_flags.extend(['--atm', args.atm])
    if args.draw:
        common_flags.extend(['--draw', args.draw])
    if args.skip_existing:
        common_flags.append('--skip_existing')
    if args.test:
        common_flags.append('--test')

    for sensor in args.sensors:
        logger.info("\n" + "#" * 80)
        logger.info(f" RUNNING EXPERIMENTS FOR SENSOR {sensor}x{sensor}")
        logger.info("#" * 80)

        sensor_flags = ['--sensor', str(sensor)] + common_flags

        # 1. Open Loop Simulation
        ol_script = os.path.join(script_dir, 'redArmSolarSCAO_01_OL.py')
        run_step([python_exec, ol_script] + sensor_flags, logger, f"OL Validation [{sensor}x{sensor}]")

        # 2. Closed Loop Baseline (delay=2)
        cl_script = os.path.join(script_dir, 'redArmSolarSCAO_02_CL_baseline.py')
        run_step([python_exec, cl_script, '--delay', '2'] + sensor_flags, logger, f"CL Baseline (delay=2) [{sensor}x{sensor}]")

        # 3a. Closed Loop POL (Linear Predictor)
        pol_script = os.path.join(script_dir, 'redArmSolarSCAO_03_CL_POL.py')
        run_step([python_exec, pol_script, '--predictor', 'linear'] + sensor_flags, logger, f"CL POL Linear [{sensor}x{sensor}]")

        # 3b. Closed Loop POL (LSTM Predictor)
        run_step([python_exec, pol_script, '--predictor', 'lstm'] + sensor_flags, logger, f"CL POL LSTM [{sensor}x{sensor}]")

        # 4. Closed Loop Sin Cerrar (Parallel Monitoring)
        sc_script = os.path.join(script_dir, 'redArmSolarSCAO_04_CL_sin_cerrar.py')
        run_step([python_exec, sc_script] + sensor_flags, logger, f"CL Sin Cerrar [{sensor}x{sensor}]")

    total_elapsed = time.time() - campaign_start_time
    logger.info("\n" + "=" * 100)
    logger.info(f" ALL SIMULATIONS COMPLETED in {total_elapsed:.2f} s ({total_elapsed/3600.0:.2f} hours)")
    logger.info("=" * 100)

    # -------------------------------------------------------------------------
    # STEP 3: RUN COMPREHENSIVE ANALYSIS REPORT
    # -------------------------------------------------------------------------
    logger.info("\n>>> [STEP 3/3] Generating Unified Analysis Report...")
    analysis_script = os.path.join(script_dir, 'analyse_comparison.py')
    subprocess.run([python_exec, analysis_script, '--base_dir', base_dir], stdout=sys.stdout, stderr=sys.stderr)

    logger.info("\n>>> CAMPAIGN COMPLETED SUCCESSFULLY!")

if __name__ == '__main__':
    main()
