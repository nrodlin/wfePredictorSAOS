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
    res_dir = os.path.join(user_home, 'simulations', 'results', 'cl_baseline')
    os.makedirs(ps_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    # Modal basis & Interaction Matrix paths
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

                res_file_name = f"res_cl_baseline_{args.delay}delay_{args.sensor}x{args.sensor}_{vibr_label}_{atm_name}_{draw_name}.h5"
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

                # Sources
                ngs = Source(magnitude=5, optBand='R4', coordinates=[0, 0], logger=logger)
                sun = ExtendedSource(optBand='V' if args.sensor == 50 else 'R', coordinates=[0, 0], nSubDirs=3, fov=9.269, subDir_margin=4.0, patch_padding=5.0, logger=logger)

                # DM
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

                # WFS & ScienceCam
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
                # WFS branch
                scao_light_path_list.append(LightPath(logger))
                scao_light_path_list[-1].initialize_path(src=sun, atm=atm, tel=est_tel, dm=dms[0], wfs=shwfs, vibration=vibrations, sci=None, delay=args.delay)

                # Science branch
                scao_light_path_list.append(LightPath(logger))
                scao_light_path_list[-1].initialize_path(src=ngs, atm=atm, tel=est_tel, dm=dms[0], wfs=None, vibration=vibrations, sci=scicam, delay=args.delay)

                lightPathTasks = [delayed(lp.propagate)(True) for lp in scao_light_path_list]

                # Interaction Matrix Handler & Controller
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
