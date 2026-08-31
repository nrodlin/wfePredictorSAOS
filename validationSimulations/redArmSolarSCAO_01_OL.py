#!/usr/bin/env python3
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import sys
import time
import json
import argparse
import logging
import numpy as np
import torch
from joblib import Parallel, delayed

from SAOS.LoggingHelper import LoggingHelper
from SAOS.ExtendedSource import ExtendedSource
from SAOS.Telescope import Telescope
from SAOS.Atmosphere import Atmosphere
from SAOS.CorrelatingShackHartmann import CorrelatingShackHartmann
from SAOS.LightPath import LightPath
from SAOS.Savepoint import Savepoint
from SAOS.Vibration import Vibration

from wfePredictorSAOS.predictor.onlinePredictor import OnlineSlopePredictor
from wfePredictorSAOS.predictor.onlineLinearPredictor import OnlineLinearSlopePredictor
from atmosphereCases import atm_cases

def parse_args():
    parser = argparse.ArgumentParser(description="OL Simulation with Linear & LSTM Predictors (2kHz)")
    parser.add_argument('--sensor', type=int, default=36, choices=[36, 50], help="Sensor grid size (36 for 36x36, 50 for 50x50)")
    parser.add_argument('--n_iterations', type=int, default=2000, help="Number of iterations (default 2000 for 1s at 2kHz)")
    parser.add_argument('--sampling_freq', type=float, default=2000.0, help="Sampling frequency in Hz (default 2000)")
    parser.add_argument('--atm', type=str, default=None, help="Atmosphere case to run (e.g. atm1). Default: all")
    parser.add_argument('--draw', type=str, default=None, help="Draw to run (e.g. draw1). Default: all")
    parser.add_argument('--no_vibr_only', action='store_true', help="Run only no-vibration cases")
    parser.add_argument('--vibr_only', action='store_true', help="Run only vibration cases")
    parser.add_argument('--generate_atm', action='store_true', help="Generate and overwrite atmosphere phase screens")
    parser.add_argument('--test', action='store_true', help="Quick test mode (50 iterations, atm1 draw1 noVibr)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.test:
        args.n_iterations = 50
        args.atm = 'atm1'
        args.draw = 'draw1'
        args.no_vibr_only = True

    test_logger = LoggingHelper(logging.INFO)
    logger = test_logger.logger

    user_home = os.path.expanduser('~')
    ps_dir = os.path.join(user_home, 'simulations', 'phase_screens')
    res_dir = os.path.join(user_home, 'simulations', 'results', 'predictor_ol')
    os.makedirs(ps_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    # Telescope setup
    diameter = 4.149
    obs_diameter = 1.3
    sampling_time = 1.0 / args.sampling_freq
    n_subaperture = args.sensor
    resolution = n_subaperture * 4
    tel_fov = 60.0

    est_tel = Telescope(
        diameter=diameter,
        resolution=resolution,
        centralObstruction=obs_diameter / diameter,
        samplingTime=sampling_time,
        fov=tel_fov,
        logger=logger
    )

    cases_to_run = {k: v for k, v in atm_cases.items() if (args.atm is None or k == args.atm)}
    
    if args.no_vibr_only:
        vibr_options = [False]
    elif args.vibr_only:
        vibr_options = [True]
    else:
        vibr_options = [False, True]

    for atm_name, draws in cases_to_run.items():
        atm_idx = int(atm_name.replace('atm', ''))
        draws_to_run = {k: v for k, v in draws.items() if (args.draw is None or k == args.draw)}

        for draw_name, kwargs in draws_to_run.items():
            t_case_start = time.time()
            logger.info(f"=== Starting OL Validation [{args.sensor}x{args.sensor}] for {atm_name} {draw_name} (r0 LOS: {kwargs['los_r0']:.2f} m) ===")

            atm_file_name = f"ps_val_{args.sensor}x{args.sensor}_{atm_name}_{draw_name}.h5"
            atm_file_path = os.path.join(ps_dir, atm_file_name)

            atm = Atmosphere(
                r0=kwargs['r0'],
                L0=kwargs['L0'],
                fractionalR0=kwargs['fractionalR0'],
                altitude=kwargs['altitude'],
                windDirection=kwargs['windDirection'],
                windSpeed=kwargs['windSpeed'],
                telescope=est_tel,
                zenith=kwargs['zenith'],
                logger=logger
            )

            if args.generate_atm or not os.path.exists(atm_file_path):
                atm.initializeAtmosphere()
                atm.save(atm_file_path)
                logger.info(f"Generated new atmosphere -> {atm_file_path}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            for use_vibrations in vibr_options:
                success = atm.load(atm_file_path)
                if not success:
                    logger.error(f"Failed to load {atm_file_path}. Skipping.")
                    continue

                if use_vibrations:
                    vibr_label = "Vibr"
                    vib_idx = atm_idx if atm_idx <= 5 else 1
                    vibration_file = os.path.join(user_home, 'simulations', 'VibrationsSource', f'EST_vibration_{vib_idx}.h5')
                    vibrations = Vibration(est_tel, vibration_file, logger)
                else:
                    vibr_label = "noVibr"
                    vibrations = None

                res_file_name = f"res_ol_{args.sensor}x{args.sensor}_{vibr_label}_{atm_name}_{draw_name}.h5"
                res_file_path = os.path.join(res_dir, res_file_name)
                savepoint = Savepoint(file_path=res_file_path, slopes=1, error=1, logger=logger)

                sun_band = 'V' if args.sensor == 50 else 'R'
                sun = ExtendedSource(
                    optBand=sun_band,
                    coordinates=[0, 0],
                    nSubDirs=3,
                    fov=9.269,
                    subDir_margin=4.0,
                    patch_padding=5.0,
                    logger=logger
                )

                shwfs = CorrelatingShackHartmann(
                    telescope=est_tel,
                    src=sun,
                    lightRatio=0.9,
                    nSubap=n_subaperture,
                    plate_scale=0.403,
                    fieldOfView=9.269,
                    guardPx=2,
                    fft_fieldOfView_oversampling=0.5,
                    use_brightest=9,
                    unit_in_rad=False,
                    logger=logger
                )

                # Build single Light Path for Open Loop
                scao_light_path_list = [LightPath(logger)]
                scao_light_path_list[0].initialize_path(
                    src=sun,
                    atm=atm,
                    tel=est_tel,
                    dm=None,
                    wfs=shwfs,
                    ncpa=None,
                    vibration=vibrations,
                    sci=None,
                    delay=0
                )

                n_slopes = scao_light_path_list[0].slopes_1D.shape[0]
                logger.info(f"Sensor {args.sensor}x{args.sensor}: Detected {n_slopes} slopes")

                predictor_lstm = OnlineSlopePredictor(
                    n_slopes=n_slopes,
                    past_horizon=24,
                    hidden_size=16,
                    n_axis=1
                )

                predictor_linear = OnlineLinearSlopePredictor(
                    n_slopes=n_slopes,
                    past_horizon=24,
                    steps_ahead=2
                )

                all_slopes_history = []
                truth_list = []
                delayed_list = []
                pred_lstm_list = []
                pred_linear_list = []

                logger.info(f"Beginning OL simulation loop ({args.n_iterations} iterations, {vibr_label})")
                for i in range(args.n_iterations):
                    if i % 200 == 0:
                        logger.info(f"Iteration {i+1}/{args.n_iterations}")

                    atm.update()
                    scao_light_path_list[0].propagate(True)
                    savepoint.save(scao_light_path_list, i)

                    current_slopes = scao_light_path_list[0].slopes_1D.copy()
                    all_slopes_history.append(current_slopes)

                    # With a 2-sample delay, at time step i the latest available measurement is from step i-2
                    if i >= 2:
                        delayed_slopes = all_slopes_history[i - 2]
                        predictor_lstm.push(delayed_slopes)
                        predictor_linear.push(delayed_slopes)

                        if predictor_lstm.ready():
                            # Predicts 2 steps ahead from delayed_slopes (i.e. targets current step i)
                            pred_lstm = predictor_lstm.predict()
                            pred_lin = predictor_linear.predict()

                            truth_list.append(current_slopes)
                            delayed_list.append(delayed_slopes)
                            pred_lstm_list.append(pred_lstm)
                            pred_linear_list.append(pred_lin)

                truth_arr = np.array(truth_list, dtype=np.float32)
                delayed_arr = np.array(delayed_list, dtype=np.float32)
                pred_lstm_arr = np.array(pred_lstm_list, dtype=np.float32)
                pred_linear_arr = np.array(pred_linear_list, dtype=np.float32)

                if len(truth_arr) > 0:
                    mse_zoh = float(np.mean((delayed_arr - truth_arr) ** 2))
                    mse_lin = float(np.mean((pred_linear_arr - truth_arr) ** 2))
                    mse_lstm = float(np.mean((pred_lstm_arr - truth_arr) ** 2))

                    rmse_zoh = float(np.sqrt(mse_zoh))
                    rmse_lin = float(np.sqrt(mse_lin))
                    rmse_lstm = float(np.sqrt(mse_lstm))

                    impr_lin = float((mse_zoh - mse_lin) / max(mse_zoh, 1e-12) * 100.0)
                    impr_lstm = float((mse_zoh - mse_lstm) / max(mse_zoh, 1e-12) * 100.0)

                    metrics = {
                        'sensor': f"{args.sensor}x{args.sensor}",
                        'atm': atm_name,
                        'draw': draw_name,
                        'vibr': vibr_label,
                        'n_samples': len(truth_arr),
                        'mse_zoh': mse_zoh,
                        'rmse_zoh': rmse_zoh,
                        'mse_linear': mse_lin,
                        'rmse_linear': rmse_lin,
                        'impr_linear_pct': impr_lin,
                        'mse_lstm': mse_lstm,
                        'rmse_lstm': rmse_lstm,
                        'impr_lstm_pct': impr_lstm
                    }

                    logger.info(f"Results for {atm_name} {draw_name} ({vibr_label}):")
                    logger.info(f"  ZOH (delay=2) RMSE: {rmse_zoh:.5f} px")
                    logger.info(f"  Linear Pred   RMSE: {rmse_lin:.5f} px (Improvement: {impr_lin:+.2f}%)")
                    logger.info(f"  LSTM Pred     RMSE: {rmse_lstm:.5f} px (Improvement: {impr_lstm:+.2f}%)")

                    # Save arrays and metrics
                    prefix = f"{args.sensor}x{args.sensor}_{vibr_label}_{atm_name}_{draw_name}"
                    np.save(os.path.join(res_dir, f"truth_{prefix}.npy"), truth_arr)
                    np.save(os.path.join(res_dir, f"pred_linear_{prefix}.npy"), pred_linear_arr)
                    np.save(os.path.join(res_dir, f"pred_lstm_{prefix}.npy"), pred_lstm_arr)
                    np.save(os.path.join(res_dir, f"delayed_{prefix}.npy"), delayed_arr)

                    with open(os.path.join(res_dir, f"metrics_{prefix}.json"), 'w') as f:
                        json.dump(metrics, f, indent=4)

            t_case_end = time.time()
            logger.info(f"=== Completed OL for {atm_name} {draw_name} in {t_case_end - t_case_start:.2f} s ===")

    test_logger = None

if __name__ == '__main__':
    main()
