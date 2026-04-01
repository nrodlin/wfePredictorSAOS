from datetime import datetime
import time
import yaml
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from joblib import Parallel, delayed

import logging
import datetime

from SAOS.LoggingHelper import LoggingHelper
from SAOS.ExtendedSource import ExtendedSource
from SAOS.Telescope import Telescope
from SAOS.Atmosphere import Atmosphere
from SAOS.DeformableMirror import DeformableMirror
from SAOS.CorrelatingShackHartmann import CorrelatingShackHartmann
from SAOS.LightPath import LightPath
from SAOS.InteractionMatrixHandler import InteractionMatrixHandler
from SAOS.Controller import Controller
from SAOS.ScienceCam import ScienceCam
from SAOS.Sharepoint import Sharepoint
from SAOS.Savepoint import Savepoint
from SAOS.Vibration import Vibration

# Logger:
test_logger = LoggingHelper(logging.INFO)

# Simulation settings:
nIterations = 1000

# Define EST
t0 = time.time()

diameter = 4.149 # in [m]
obs_diameter = 1.3 # in [m]
sampling_time = 1/2000 # in [s]
n_subaperture = 36
resolution = n_subaperture * 4 # resolution of the phase screen in [px]
pixel_size = diameter / resolution
tel_fov = 60 # in [arcsec]

est_tel = Telescope(diameter = diameter,
                    resolution = resolution,
                    centralObstruction= obs_diameter / diameter,
                    samplingTime=sampling_time,
                    fov=tel_fov,
                    logger=test_logger.logger)

with open('/home/nlinares/code/wfePredictorSAOS/trainingDataSimulations/atmosphereCases.yaml', 'r') as f:
    atm_cases = yaml.safe_load(f)

for atm_name, draws in atm_cases.items():
    atm_num = atm_name.replace('atm', '')
    for draw_name, kwargs in draws.items():
        draw_num = draw_name.replace('draw', '')
        
        user_home = os.path.expanduser('~')
        load_filename_atm = f'{user_home}/simulations/phase_screens/ps_atm{atm_num}_draw{draw_num}.h5'
        
        vib_idx = int(atm_num)
        test_logger.logger.info(f'--- Starting Simulation for {atm_name} {draw_name} with Vibration {vib_idx} ---')
        t_case_start = time.time()

        # Define the savingpoint
        savepoint = Savepoint(file_path=f"{user_home}/simulations/results/res_atm{atm_num}_draw{draw_num}_vib{vib_idx}.h5", slopes=1, error=1, logger=test_logger.logger)
        
        # Atmosphere:
        atm = Atmosphere(r0=kwargs['r0'],
                         L0=kwargs['L0'],
                         fractionalR0=kwargs['fractionalR0'],
                         altitude=kwargs['altitude'],
                         windDirection=kwargs['windDirection'],
                         windSpeed=kwargs['windSpeed'],
                         telescope=est_tel,
                         zenith=kwargs['zenith'],
                         logger=test_logger.logger)

        success = atm.load(load_filename_atm)
        if not success:
           test_logger.logger.error(f"Atmosphere file {load_filename_atm} could not be loaded")
           continue
           
        vibration_file = f'{user_home}/simulations/VibrationsSource/EST_vibration_{vib_idx}.h5'
        vibrations = Vibration(est_tel, vibration_file, test_logger.logger)

        # Sources:
        sun = ExtendedSource(optBand='R',
                             coordinates=[0, 0],
                             nSubDirs=3,
                             fov=9.269,
                             subDir_margin=4.0,
                             patch_padding=5.0,
                             logger=test_logger.logger)

        # Wavefront Sensor
        shwfs_0 = CorrelatingShackHartmann(telescope=est_tel,
                                           src=sun,
                                           lightRatio=0.9,
                                           nSubap=n_subaperture,
                                           plate_scale=0.403,
                                           fieldOfView=9.269,
                                           guardPx=2,
                                           fft_fieldOfView_oversampling=0.5,
                                           use_brightest=9,
                                           unit_in_rad=False,
                                           logger=test_logger.logger)

        # Build the Light Path
        scao_light_path_list = []
        # Create red branch with 0 delay samples
        scao_light_path_list.append(LightPath(test_logger.logger))
        scao_light_path_list[-1].initialize_path(src=sun, atm=atm, tel=est_tel, dm=None, wfs=shwfs_0, ncpa=None, vibration=vibrations, sci=None, delay=0)

        # Create red branch with 2 delay samples
        scao_light_path_list.append(LightPath(test_logger.logger))
        scao_light_path_list[-1].initialize_path(src=sun, atm=atm, tel=est_tel, dm=None, wfs=shwfs_0, ncpa=None, vibration=vibrations, sci=None, delay=2)

        lightPathTasks = []
        for i in range(len(scao_light_path_list)):
            lightPathTasks.append(delayed(scao_light_path_list[i].propagate)(True))

        test_logger.logger.info(f'Beginning SCAO loop for {atm_name} {draw_name} vib {vib_idx}')

        # SCAO loop
        for i in range(nIterations):
            if i % 100 == 0:
                est_tel.logger.info(f'Iteration {i+1}')
            # Update the atmosphere
            atm.update()
            # Propagate the light
            Parallel(n_jobs=2, prefer='threads')(lightPathTasks)
            
            # Save Data
            savepoint.save(scao_light_path_list, i)

        test_logger.logger.info(f'Simulation ended for {atm_name} {draw_name} vib {vib_idx}.')
        t_case_end = time.time()
        test_logger.logger.info(f'Elapsed time for {atm_name} {draw_name} vib {vib_idx}: {t_case_end - t_case_start:.2f} [s]')

# Force destructor call for the qeue of logs
test_logger = None
