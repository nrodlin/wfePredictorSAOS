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
from SAOS.Source import Source
from SAOS.ExtendedSource import ExtendedSource
from SAOS.Telescope import Telescope
from SAOS.Atmosphere import Atmosphere
from SAOS.DeformableMirror import DeformableMirror
from SAOS.Vibration import Vibration
from SAOS.CorrelatingShackHartmann import CorrelatingShackHartmann
from SAOS.LightPath import LightPath
from SAOS.InteractionMatrixHandler import InteractionMatrixHandler
from SAOS.Controller import Controller
from SAOS.ScienceCam import ScienceCam
from SAOS.Savepoint import Savepoint

from wfePredictorSAOS.predictor.onlinePredictor import OnlineSlopePredictor
from wfePredictorSAOS.predictor.onlineLinearPredictor import OnlineLinearSlopePredictor
from atmosphereCases import atm_cases

def parse_args():
    parser = argparse.ArgumentParser(description="Closed-Loop with Open Predictor (CL + Sin Cerrar) Simulation (2kHz)")
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
    res_dir = os.path.join(user_home, 'simulations', 'results', 'cl_sin_cerrar')
    os.makedirs(ps_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    if args.sensor == 36:
        load_filename_modalBasis = os.path.join(user_home, 'simulations', 'modal_basis', 'predictor_modalBasis.h5')
        load_filename_IM = os.path.join(user_home, 'simulations', 'interaction_matrix', 'predictor_IM.h5')
    else:
        load_filename_modalBasis = os.path.join(user_home, 'simulations', 'modal_basis', 'predictor_50x50_modalBasis.h5')
        load_filename_IM = os.path.join(user_home, 'simulations', 'interaction_matrix', 'predictor_50x50_IM.h5')

    diameter = 4.149
    obs_diameter = 1.3
    sampling_time = 1.0 / args.sampling_freq
    n_subaperture = args.sensor
    resolution = n_subaperture * 4
    tel_fov = 60.0
    scienceFs = 56.0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
            logger.info(f"=== Starting CL Sin Cerrar [{args.sensor}x{args.sensor}] for {atm_name} {draw_name} ===")

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

                res_file_name = f"res_cl_sin_cerrar_{args.sensor}x{args.sensor}_{vibr_label}_{atm_name}_{draw_name}.h5"
                res_file_path = os.path.join(res_dir, res_file_name)
                savepoint = Savepoint(
                    file_path=res_file_path,
                    atm=1,
                    dm=1,
                    slopes=1,
                    sci=1,
                    sci_frame=1,
                    only_metrics=1,
                    logger=logger
                )

                ngs = Source(magnitude=5, optBand='R4', coordinates=[0, 0], logger=logger)
                sun = ExtendedSource(
                    optBand='V' if args.sensor == 50 else 'R',
                    coordinates=[0, 0],
                    nSubDirs=3,
                    fov=9.269,
                    subDir_margin=4.0,
                    patch_padding=5.0,
                    logger=logger
                )

                asm_params = {'dynamicModel': os.path.join(user_home, 'simulations', 'MirrorModels', 'asm_discrete_model.h5'), 'validActThreshpercentage': 0.5}
                asm = DeformableMirror(
                    telescope=est_tel,
                    nActs=n_subaperture + 1,
                    altitude=0,
                    typeDM='cartesian',
                    logger=logger,
                    **asm_params
                )
                dms = [asm]

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

                scicam = ScienceCam(
                    fieldOfView=9.269,
                    plate_scale=0.0167,
                    samplingTime=est_tel.samplingTime,
                    telescope=est_tel,
                    integrationTime=1.0 / scienceFs,
                    noiseFlag=False,
                    logger=logger
                )

                scao_light_path_list = []
                scao_light_path_list.append(LightPath(logger))
                scao_light_path_list[-1].initialize_path(src=sun, atm=atm, tel=est_tel, dm=dms[0], wfs=shwfs, vibration=vibrations, sci=None, delay=2)

                scao_light_path_list.append(LightPath(logger))
                scao_light_path_list[-1].initialize_path(src=ngs, atm=atm, tel=est_tel, dm=dms[0], wfs=None, vibration=vibrations, sci=scicam, delay=2)

                lightPathTasks = [delayed(lp.propagate)(True) for lp in scao_light_path_list]

                im_handler = InteractionMatrixHandler(logger)
                im_handler.initialize_im_class(scao_light_path_list)
                im_handler.load_modalBasis(load_filename_modalBasis)
                im_handler.load_IM(load_filename_IM)

                controller_kwargs = {
                    'rcond': 0.025,
                    'beta': 1e-4,
                    'gain': [0.25],
                    'decay': [0.999],
                    'ki': [0.0]
                }

                controller = Controller(
                    telescope=est_tel,
                    interactionMatrix=im_handler,
                    reconstructionMethod='inversion',
                    controllerType='leaky',
                    logger=logger,
                    **controller_kwargs
                )

                n_slopes = scao_light_path_list[0].slopes_1D.shape[0]
                n_modes = controller.reconstructor[0].shape[0]

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

                im_tensor = torch.as_tensor(im_handler.interaction_matrix_warehouse[0][0]['IM'], dtype=torch.float64, device=device)
                cmd_history = [torch.zeros((n_modes, 1), dtype=torch.float64, device=device) for _ in range(4)]

                slopes_res_list = []
                slopes_pol_list = []
                pred_lstm_list = []
                pred_linear_list = []

                logger.info(f"Beginning CL Sin Cerrar loop ({args.n_iterations} iterations, {vibr_label})")
                for i in range(args.n_iterations):
                    if i % 200 == 0:
                        logger.info(f"Iteration {i+1}/{args.n_iterations}")

                    atm.update()
                    Parallel(n_jobs=1, prefer="threads")(lightPathTasks)

                    # Standard closed loop computes control action & updates DM
                    cmd = controller.computeControlAction(scao_light_path_list)
                    for j in range(len(dms)):
                        dms[j].updateDMShape(cmd[j])

                    # Current modal command from controller
                    curr_modal_cmd = controller.command_previous[0].to(device)

                    # Residual slopes at t-2
                    res_slopes = scao_light_path_list[0].slopes_1D.copy()
                    res_slopes_tensor = torch.as_tensor(res_slopes, dtype=torch.float64, device=device).unsqueeze(1)

                    # Delayed modal command applied 2 samples ago
                    cmd_delayed_2 = cmd_history[-2]

                    # Reconstructed POL slopes: s_pol = s_res + IM @ cmd(t-2)
                    pol_slopes_tensor = res_slopes_tensor + im_tensor @ cmd_delayed_2
                    pol_slopes = pol_slopes_tensor.squeeze(1).cpu().numpy()

                    # Push POL slopes to predictors
                    predictor_lstm.push(pol_slopes)
                    predictor_linear.push(pol_slopes)

                    if predictor_lstm.ready():
                        pred_lstm = predictor_lstm.predict()
                        pred_lin = predictor_linear.predict()

                        slopes_res_list.append(res_slopes)
                        slopes_pol_list.append(pol_slopes)
                        pred_lstm_list.append(pred_lstm)
                        pred_linear_list.append(pred_lin)

                    # Update modal command history
                    cmd_history.pop(0)
                    cmd_history.append(curr_modal_cmd.clone())

                    savepoint.save([atm], i)
                    savepoint.save(dms, i)
                    savepoint.save(scao_light_path_list, i)

                res_arr = np.array(slopes_res_list, dtype=np.float32)
                pol_arr = np.array(slopes_pol_list, dtype=np.float32)
                pred_lstm_arr = np.array(pred_lstm_list, dtype=np.float32)
                pred_linear_arr = np.array(pred_linear_list, dtype=np.float32)

                if len(pol_arr) > 2:
                    # In POL prediction, pred at step k predicts POL slopes at step k+2
                    target_pol = pol_arr[2:]
                    eval_pred_lstm = pred_lstm_arr[:-2]
                    eval_pred_linear = pred_linear_arr[:-2]
                    eval_zoh = pol_arr[:-2]

                    mse_zoh = float(np.mean((eval_zoh - target_pol) ** 2))
                    mse_lin = float(np.mean((eval_pred_linear - target_pol) ** 2))
                    mse_lstm = float(np.mean((eval_pred_lstm - target_pol) ** 2))

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
                        'n_samples': len(target_pol),
                        'mse_pol_zoh': mse_zoh,
                        'rmse_pol_zoh': rmse_zoh,
                        'mse_pol_linear': mse_lin,
                        'rmse_pol_linear': rmse_lin,
                        'impr_linear_pct': impr_lin,
                        'mse_pol_lstm': mse_lstm,
                        'rmse_pol_lstm': rmse_lstm,
                        'impr_lstm_pct': impr_lstm
                    }

                    logger.info(f"Results for CL Sin Cerrar {atm_name} {draw_name} ({vibr_label}):")
                    logger.info(f"  POL ZOH (delay=2) RMSE: {rmse_zoh:.5f} px")
                    logger.info(f"  POL Linear Pred   RMSE: {rmse_lin:.5f} px (Improvement: {impr_lin:+.2f}%)")
                    logger.info(f"  POL LSTM Pred     RMSE: {rmse_lstm:.5f} px (Improvement: {impr_lstm:+.2f}%)")

                    prefix = f"{args.sensor}x{args.sensor}_{vibr_label}_{atm_name}_{draw_name}"
                    np.save(os.path.join(res_dir, f"slopes_res_{prefix}.npy"), res_arr)
                    np.save(os.path.join(res_dir, f"slopes_pol_{prefix}.npy"), pol_arr)
                    np.save(os.path.join(res_dir, f"pred_pol_linear_{prefix}.npy"), pred_linear_arr)
                    np.save(os.path.join(res_dir, f"pred_pol_lstm_{prefix}.npy"), pred_lstm_arr)

                    with open(os.path.join(res_dir, f"metrics_{prefix}.json"), 'w') as f:
                        json.dump(metrics, f, indent=4)

            t_case_end = time.time()
            logger.info(f"=== Completed CL Sin Cerrar for {atm_name} {draw_name} in {t_case_end - t_case_start:.2f} s ===")

    test_logger = None

if __name__ == '__main__':
    main()
