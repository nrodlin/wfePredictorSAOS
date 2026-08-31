#!/usr/bin/env python3
"""
Pre-generation script for AO Predictor Assets:
- Atmosphere phase screens (15 cases: atm1..5 x draw1..3)
- Modal Bases (36x36 ASM radial & 50x50 M7 cartesian)
- Interaction Matrices (36x36 & 50x50)

Can be executed standalone or called before launching the simulation campaign.
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import sys
import time
import argparse
import logging
import numpy as np
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from SAOS.LoggingHelper import LoggingHelper
from SAOS.Source import Source
from SAOS.ExtendedSource import ExtendedSource
from SAOS.Telescope import Telescope
from SAOS.Atmosphere import Atmosphere
from SAOS.DeformableMirror import DeformableMirror
from SAOS.CorrelatingShackHartmann import CorrelatingShackHartmann
from SAOS.LightPath import LightPath
from SAOS.InteractionMatrixHandler import InteractionMatrixHandler
from SAOS.ScienceCam import ScienceCam

try:
    from atmosphereCases import atm_cases
except ImportError:
    from validationSimulations.atmosphereCases import atm_cases

def get_base_dir(custom_base_dir=None):
    if custom_base_dir:
        return custom_base_dir
    # Check if NAS is mounted (e.g. on cluster node durum)
    if os.path.exists('/mnt/nas-mcao'):
        nas_pred_dir = '/mnt/nas-mcao/predictor_sims'
        os.makedirs(nas_pred_dir, exist_ok=True)
        return nas_pred_dir
    return os.path.join(os.path.expanduser('~'), 'simulations')

def get_asset_source_dirs(base_dir):
    user_home_sims = os.path.join(os.path.expanduser('~'), 'simulations')
    
    # MirrorModels
    if os.path.exists(os.path.join(base_dir, 'MirrorModels')):
        mirror_models_dir = os.path.join(base_dir, 'MirrorModels')
    else:
        mirror_models_dir = os.path.join(user_home_sims, 'MirrorModels')
        
    # VibrationsSource
    if os.path.exists(os.path.join(base_dir, 'VibrationsSource')):
        vibrations_dir = os.path.join(base_dir, 'VibrationsSource')
    else:
        vibrations_dir = os.path.join(user_home_sims, 'VibrationsSource')
        
    return mirror_models_dir, vibrations_dir

def parse_args():
    parser = argparse.ArgumentParser(description="Pre-generate Phase Screens, Modal Bases, and Interaction Matrices")
    parser.add_argument('--base_dir', type=str, default=None, help="Base directory for simulations (default: /mnt/nas-mcao/predictor_sims or ~/simulations)")
    parser.add_argument('--force_atm', action='store_true', help="Force re-generating phase screens even if they exist")
    parser.add_argument('--force_modal', action='store_true', help="Force re-generating modal bases even if they exist")
    parser.add_argument('--force_im', action='store_true', help="Force re-measuring interaction matrices even if they exist")
    return parser.parse_args()

def prepare_all_assets(base_dir=None, force_atm=False, force_modal=False, force_im=False, logger=None):
    if logger is None:
        test_logger = LoggingHelper(logging.INFO)
        logger = test_logger.logger

    base_dir = get_base_dir(base_dir)
    mirror_models_dir, vibrations_dir = get_asset_source_dirs(base_dir)

    ps_dir = os.path.join(base_dir, 'phase_screens')
    mb_dir = os.path.join(base_dir, 'modal_basis')
    im_dir = os.path.join(base_dir, 'interaction_matrix')
    res_dir = os.path.join(base_dir, 'results')

    os.makedirs(ps_dir, exist_ok=True)
    os.makedirs(mb_dir, exist_ok=True)
    os.makedirs(im_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    logger.info(f"==============================================================================")
    logger.info(f" PRE-GENERATING ASSETS FOR AO PREDICTOR (Durum / Local)")
    logger.info(f" Base Directory: {base_dir}")
    logger.info(f" Phase Screens Dir: {ps_dir}")
    logger.info(f" Modal Basis Dir:   {mb_dir}")
    logger.info(f" IM Dir:            {im_dir}")
    logger.info(f"==============================================================================")

    # 1. Telescope
    diameter = 4.149
    obs_diameter = 1.3
    sampling_time = 1.0 / 2000.0
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

    # =========================================================================
    # [1] PRE-GENERATE ATMOSPHERIC PHASE SCREENS
    # =========================================================================
    logger.info("\n--- [1/3] Checking Atmospheric Phase Screens (15 cases) ---")
    n_generated_atm = 0
    for atm_name, draws in atm_cases.items():
        atm_idx = int(atm_name.replace('atm', ''))
        for draw_name, kwargs in draws.items():
            atm_file_name = f"ps_dualARM_EST_case{atm_idx}_{draw_name}.h5"
            atm_file_path = os.path.join(ps_dir, atm_file_name)

            if not force_atm and os.path.exists(atm_file_path):
                logger.info(f"Atmosphere screen already exists: {atm_file_name} -> OK")
                continue

            logger.info(f"Generating atmosphere screen: {atm_file_name} (r0_los={kwargs['los_r0']}m)...")
            t_start = time.time()
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
            if os.path.exists(atm_file_path):
                os.remove(atm_file_path)
            atm.initializeAtmosphere()
            atm.save(atm_file_path)
            n_generated_atm += 1
            logger.info(f"Generated and saved {atm_file_name} in {time.time() - t_start:.2f} s")

    logger.info(f"Atmosphere check completed. ({n_generated_atm} newly generated)")

    # =========================================================================
    # [2] PRE-GENERATE / CHECK MODAL BASES AND INTERACTION MATRICES
    # =========================================================================
    logger.info("\n--- [2/3] Checking Modal Bases & [3/3] Interaction Matrices ---")
    
    # Dummy atmosphere for IM setup
    atm_dummy = Atmosphere(
        r0=0.15, L0=25, fractionalR0=[0.55, 0.2, 0.12, 0.08, 0.05],
        altitude=[100, 1500, 5000, 10000, 15000],
        windDirection=[20, 70, 120, 210, 300],
        windSpeed=[8, 12, 16, 11, 7],
        telescope=est_tel, zenith=30, logger=logger
    )

    sensor_configs = [
        {
            'sensor': 36,
            'name': '36x36 (ASM 37x37 Radial)',
            'mb_file': os.path.join(mb_dir, 'predictor_36x36_modalBasis.h5'),
            'im_file': os.path.join(im_dir, 'predictor_36x36_IM.h5'),
            'typeDM': 'radial',
            'nActs': 37,
            'validActThreshpercentage': 0.7533,
            'dynamicModel': os.path.join(mirror_models_dir, 'asm_discrete_model.h5'),
            'optBand': 'R',
            'fov': 9.269,
            'plate_scale': 0.403,
            'nModes': 500,
            'stroke': 8e-7
        },
        {
            'sensor': 50,
            'name': '50x50 (M7 51x51 Cartesian)',
            'mb_file': os.path.join(mb_dir, 'predictor_50x50_modalBasis.h5'),
            'im_file': os.path.join(im_dir, 'predictor_50x50_IM.h5'),
            'typeDM': 'cartesian',
            'nActs': 51,
            'validActThreshpercentage': 0.5,
            'dynamicModel': os.path.join(mirror_models_dir, 'm7_discrete_model.h5'),
            'optBand': 'V',
            'fov': 9.975,
            'plate_scale': 0.475,
            'nModes': 1077,
            'stroke': 9e-7
        }
    ]

    for cfg in sensor_configs:
        logger.info(f"\nProcessing Sensor {cfg['name']}...")
        mb_file = cfg['mb_file']
        im_file = cfg['im_file']

        mb_exists = os.path.exists(mb_file) and not force_modal
        im_exists = os.path.exists(im_file) and not force_im

        if mb_exists and im_exists:
            logger.info(f"Modal basis ({os.path.basename(mb_file)}) and IM ({os.path.basename(im_file)}) already exist -> OK")
            continue

        # Initialize DM and WFS for this sensor
        sun = ExtendedSource(
            optBand=cfg['optBand'], coordinates=[0, 0], nSubDirs=3,
            fov=cfg['fov'], subDir_margin=4.0, patch_padding=5.0, logger=logger
        )
        dm = DeformableMirror(
            telescope=est_tel, nActs=cfg['nActs'], altitude=0,
            typeDM=cfg['typeDM'], logger=logger,
            validActThreshpercentage=cfg['validActThreshpercentage'],
            dynamicModel=cfg['dynamicModel']
        )
        wfs = CorrelatingShackHartmann(
            telescope=est_tel, src=sun, lightRatio=0.9, nSubap=cfg['sensor'],
            plate_scale=cfg['plate_scale'], fieldOfView=cfg['fov'], guardPx=2,
            fft_fieldOfView_oversampling=0.5, use_brightest=9, unit_in_rad=False, logger=logger
        )
        scicam = ScienceCam(
            fieldOfView=cfg['fov'], plate_scale=0.015, samplingTime=est_tel.samplingTime,
            telescope=est_tel, integrationTime=1./56., noiseFlag=False, logger=logger
        )

        lp = LightPath(logger)
        lp.initialize_path(src=sun, atm=atm_dummy, tel=est_tel, dm=dm, wfs=wfs, vibration=None, sci=scicam, delay=1)

        im_handler = InteractionMatrixHandler(logger)
        im_handler.initialize_im_class([lp])

        # Modal basis
        if mb_exists:
            logger.info(f"Loading existing modal basis from {mb_file}")
            im_handler.load_modalBasis(mb_file)
        else:
            logger.info(f"Generating new modal basis for {cfg['name']}...")
            t_start = time.time()
            im_handler.generate_modal_basis()
            im_handler.save_modalBasis(mb_file)
            logger.info(f"Modal basis saved to {mb_file} in {time.time() - t_start:.2f} s")

        # Interaction Matrix
        if im_exists:
            logger.info(f"Loading existing IM from {im_file}")
            im_handler.load_IM(im_file)
        else:
            logger.info(f"Measuring new Interaction Matrix for {cfg['name']} (stroke={cfg['stroke']}, nModes={cfg['nModes']})...")
            t_start = time.time()
            im_handler.measure(modal_basis='zernike', stroke=cfg['stroke'], nModes=[cfg['nModes']])
            im_handler.save_IM(im_file)
            logger.info(f"Interaction Matrix saved to {im_file} in {time.time() - t_start:.2f} s")

    logger.info(f"\n==============================================================================")
    logger.info(f" ALL ASSETS READY FOR SIMULATION CAMPAIGN!")
    logger.info(f"==============================================================================")

def main():
    args = parse_args()
    prepare_all_assets(
        base_dir=args.base_dir,
        force_atm=args.force_atm,
        force_modal=args.force_modal,
        force_im=args.force_im
    )

if __name__ == '__main__':
    main()
