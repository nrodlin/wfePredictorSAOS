#!/usr/bin/env python3
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import sys
import time
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

from atmosphereCases import atm_cases

def parse_args():
    parser = argparse.ArgumentParser(description="Closed-Loop Baseline Simulation (2kHz)")
    parser.add_argument('--sensor', type=int, default=36, choices=[36, 50], help="Sensor grid size (36 for 36x36, 50 for 50x50)")
    parser.add_argument('--delay', type=int, default=2, help="Loop delay samples (default 2)")
    parser.add_argument('--n_iterations', type=int, default=2500, help="Number of iterations (default 2500 for 1.25s at 2kHz)")
    parser.add_argument('--sampling_freq', type=float, default=2000.0, help="Sampling frequency in Hz (default 2000)")
    parser.add_argument('--gain', type=float, default=0.25, help="Loop gain (default 0.25)")
    parser.add_argument('--decay', type=float, default=0.999, help="Leaky decay factor (default 0.999)")
    parser.add_argument('--atm', type=str, default=None, help="Atmosphere case to run (e.g. atm1). Default: all")
    parser.add_argument('--draw', type=str, default=None, help="Draw to run (e.g. draw1). Default: all")
    parser.add_argument('--no_vibr_only', action='store_true', help="Run only no-vibration cases")
    parser.add_argument('--vibr_only', action='store_true', help="Run only vibration cases")
    parser.add_argument('--generate_atm', action='store_true', help="Generate and overwrite atmosphere phase screens")
    parser.add_argument('--base_dir', type=str, default=None, help="Base directory for simulations (default: /mnt/nas-mcao/predictor_sims or ~/simulations)")
    parser.add_argument('--skip_existing', action='store_true', help="Skip simulation if result file already exists")
    parser.add_argument('--test', action='store_true', help="Quick test mode (50 iterations, atm1 draw1 noVibr)")
    return parser.parse_args()

def get_base_dir(custom_base_dir=None):
    if custom_base_dir:
        return custom_base_dir
    if os.path.exists('/mnt/nas-mcao'):
        nas_pred_dir = '/mnt/nas-mcao/predictor_sims'
        os.makedirs(nas_pred_dir, exist_ok=True)
        return nas_pred_dir
    return os.path.join(os.path.expanduser('~'), 'simulations')

def get_asset_dirs(base_dir):
    user_home_sims = os.path.join(os.path.expanduser('~'), 'simulations')
    mirror_models_dir = os.path.join(base_dir, 'MirrorModels') if os.path.exists(os.path.join(base_dir, 'MirrorModels')) else os.path.join(user_home_sims, 'MirrorModels')
    vibrations_dir = os.path.join(base_dir, 'VibrationsSource') if os.path.exists(os.path.join(base_dir, 'VibrationsSource')) else os.path.join(user_home_sims, 'VibrationsSource')
    return mirror_models_dir, vibrations_dir

def main():
    args = parse_args()

    if args.test:
        if args.n_iterations == 2500:
            args.n_iterations = 50
        args.atm = 'atm1'
        args.draw = 'draw1'
        args.no_vibr_only = True

    test_logger = LoggingHelper(logging.INFO)
    logger = test_logger.logger

    base_dir = get_base_dir(args.base_dir)
    mirror_models_dir, vibrations_dir = get_asset_dirs(base_dir)

    ps_dir = os.path.join(base_dir, 'phase_screens')
    res_dir = os.path.join(base_dir, 'results', 'cl_baseline')
    os.makedirs(ps_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    # Modal basis & Interaction Matrix paths
    if args.sensor == 36:
        load_filename_modalBasis = os.path.join(base_dir, 'modal_basis', 'predictor_36x36_modalBasis.h5')
        load_filename_IM = os.path.join(base_dir, 'interaction_matrix', 'predictor_36x36_IM.h5')
    else:
        load_filename_modalBasis = os.path.join(base_dir, 'modal_basis', 'predictor_50x50_modalBasis.h5')
        load_filename_IM = os.path.join(base_dir, 'interaction_matrix', 'predictor_50x50_IM.h5')

    diameter = 4.149
    obs_diameter = 1.3
    sampling_time = 1.0 / args.sampling_freq
    n_subaperture = args.sensor
    resolution = 200
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
            logger.info(f"=== Starting CL Baseline [{args.sensor}x{args.sensor}, delay={args.delay}] for {atm_name} {draw_name} ===")

            atm_file_name = f"ps_dualARM_EST_case{atm_idx}_{draw_name}.h5"
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
                if os.path.exists(atm_file_path):
                    os.remove(atm_file_path)
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
                    vibration_file = os.path.join(vibrations_dir, f'EST_vibration_{vib_idx}.h5')
                    vibrations = Vibration(est_tel, vibration_file, logger)
                else:
                    vibr_label = "noVibr"
                    vibrations = None

                res_file_name = f"res_cl_baseline_{args.delay}delay_{args.sensor}x{args.sensor}_{vibr_label}_{atm_name}_{draw_name}.h5"
                res_file_path = os.path.join(res_dir, res_file_name)

                if args.skip_existing and os.path.exists(res_file_path):
                    logger.info(f"Skipping {res_file_name} - already exists at {res_file_path}")
                    continue

                if os.path.exists(res_file_path):
                    os.remove(res_file_path)
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

                # Sources & DM geometry configured according to pdrClosure
                if args.sensor == 36:
                    sun = ExtendedSource(optBand='R', coordinates=[0, 0], nSubDirs=3, fov=9.269, subDir_margin=4.0, patch_padding=5.0, logger=logger)
                    ngs = Source(magnitude=5, optBand='R4', coordinates=[0, 0], logger=logger)
                    dm_params = {'dynamicModel': os.path.join(mirror_models_dir, 'asm_discrete_model.h5'), 'validActThreshpercentage': 0.7533}
                    dm = DeformableMirror(telescope=est_tel, nActs=37, altitude=0, typeDM='radial', logger=logger, **dm_params)
                    wfs_plate_scale = 0.403
                    wfs_fov = 9.269
                    sci_fov = 9.269
                    sci_plate_scale = 0.0167
                else:
                    sun = ExtendedSource(optBand='V', coordinates=[0, 0], nSubDirs=3, fov=9.975, subDir_margin=4.0, patch_padding=5.0, logger=logger)
                    ngs = Source(magnitude=5, optBand='V0', coordinates=[0, 0], logger=logger)
                    dm_params = {'dynamicModel': os.path.join(mirror_models_dir, 'm7_discrete_model.h5'), 'validActThreshpercentage': 0.5}
                    dm = DeformableMirror(telescope=est_tel, nActs=51, altitude=0, typeDM='cartesian', logger=logger, **dm_params)
                    wfs_plate_scale = 0.475
                    wfs_fov = 9.975
                    sci_fov = 9.975
                    sci_plate_scale = 0.0128

                dms = [dm]

                # WFS
                shwfs = CorrelatingShackHartmann(
                    telescope=est_tel,
                    src=sun,
                    lightRatio=0.9,
                    nSubap=n_subaperture,
                    plate_scale=wfs_plate_scale,
                    fieldOfView=wfs_fov,
                    guardPx=2,
                    fft_fieldOfView_oversampling=0.5,
                    use_brightest=9,
                    unit_in_rad=False,
                    logger=logger
                )

                # Science Cameras: Standard Long Exposure (56 Hz) and Ultra Long Exposure (5 Hz)
                scicam_56 = ScienceCam(
                    fieldOfView=sci_fov,
                    plate_scale=sci_plate_scale,
                    samplingTime=est_tel.samplingTime,
                    telescope=est_tel,
                    integrationTime=1.0 / 56.0,
                    noiseFlag=False,
                    logger=logger
                )
                scicam_5 = ScienceCam(
                    fieldOfView=sci_fov,
                    plate_scale=sci_plate_scale,
                    samplingTime=est_tel.samplingTime,
                    telescope=est_tel,
                    integrationTime=1.0 / 5.0,
                    noiseFlag=False,
                    logger=logger
                )

                scao_light_path_list = []
                # WFS branch (LP0)
                scao_light_path_list.append(LightPath(logger))
                scao_light_path_list[-1].initialize_path(src=sun, atm=atm, tel=est_tel, dm=dms[0], wfs=shwfs, vibration=vibrations, sci=None, delay=args.delay)

                # Science branch 56 Hz (LP1)
                scao_light_path_list.append(LightPath(logger))
                scao_light_path_list[-1].initialize_path(src=ngs, atm=atm, tel=est_tel, dm=dms[0], wfs=None, vibration=vibrations, sci=scicam_56, delay=args.delay)

                # Science branch 5 Hz (LP2)
                scao_light_path_list.append(LightPath(logger))
                scao_light_path_list[-1].initialize_path(src=ngs, atm=atm, tel=est_tel, dm=dms[0], wfs=None, vibration=vibrations, sci=scicam_5, delay=args.delay)

                lightPathTasks = [delayed(lp.propagate)(True) for lp in scao_light_path_list]

                # Interaction Matrix Handler & Controller
                im_handler = InteractionMatrixHandler(logger)
                im_handler.initialize_im_class(scao_light_path_list)
                im_handler.load_modalBasis(load_filename_modalBasis)
                im_handler.load_IM(load_filename_IM)

                controller_kwargs = {
                    'rcond': 0.025,
                    'beta': 1e-4,
                    'gain': [args.gain],
                    'decay': [args.decay],
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

                logger.info(f"Beginning CL Baseline loop ({args.n_iterations} iterations, {vibr_label})")
                for i in range(args.n_iterations):
                    if i % 200 == 0:
                        logger.info(f"Iteration {i+1}/{args.n_iterations}")

                    atm.update()
                    Parallel(n_jobs=1, prefer="threads")(lightPathTasks)

                    cmd = controller.computeControlAction(scao_light_path_list)
                    for j in range(len(dms)):
                        dms[j].updateDMShape(cmd[j])

                    savepoint.save([atm], i)
                    savepoint.save(dms, i)
                    savepoint.save(scao_light_path_list, i)

                logger.info(f"Simulation ended for {atm_name} {draw_name} ({vibr_label})")

            t_case_end = time.time()
            logger.info(f"=== Completed CL Baseline for {atm_name} {draw_name} in {t_case_end - t_case_start:.2f} s ===")

    test_logger = None

if __name__ == '__main__':
    main()
