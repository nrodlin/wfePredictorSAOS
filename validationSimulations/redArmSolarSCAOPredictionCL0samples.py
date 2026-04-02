from datetime import datetime
import time
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from joblib import Parallel, delayed
import logging

from SAOS.LoggingHelper import LoggingHelper
from SAOS.Source import Source
from SAOS.ExtendedSource import ExtendedSource
from SAOS.Telescope import Telescope
from SAOS.Atmosphere import Atmosphere
from SAOS.DeformableMirror import DeformableMirror
from SAOS.Vibration import Vibration
from SAOS.ShackHartmann import ShackHartmann
from SAOS.CorrelatingShackHartmann import CorrelatingShackHartmann
from SAOS.LightPath import LightPath
from SAOS.InteractionMatrixHandler import InteractionMatrixHandler
from SAOS.Controller import Controller
from SAOS.ScienceCam import ScienceCam
from SAOS.Sharepoint import Sharepoint
from SAOS.Savepoint import Savepoint

from atmosphereCases import atm_cases

# Logger:
test_logger = LoggingHelper(logging.INFO)

# Simulation settings:
nIterations = 2000
scienceFs = 56. # Hz

spider = False
measure_new_IM = False
load_modal_basis = True

nModes = None # [nModesASM]
im_stroke = [5e-7] # in meters

user_home = os.path.expanduser('~')
ps_dir = os.path.join(user_home, 'simulations', 'phase_screens')
res_dir = os.path.join(user_home, 'simulations', 'results', 'cl_0samples')

os.makedirs(res_dir, exist_ok=True)

# Loading files:
load_filename_modalBasis = os.path.join(user_home, 'simulations', 'modal_basis', 'predictor_modalBasis.h5')
load_filename_IM = os.path.join(user_home, 'simulations', 'interaction_matrix', 'predictor_IM.h5')

start_atm_idx = None # None to start from beginning

## Define EST
t0 = time.time()

diameter = 4.149 # in [m]
obs_diameter = 1.3 # in [m]
sampling_time = 1/2000 # in [s]
n_subaperture_red = 36
resolution = n_subaperture_red * 4 # resolution of the phase screen in [px]
pixel_size = diameter / resolution
tel_fov = 60 # in [arcsec]

est_tel = Telescope(diameter=diameter,
                    resolution=resolution,
                    centralObstruction=obs_diameter / diameter,
                    samplingTime=sampling_time,
                    fov=tel_fov,
                    logger=test_logger.logger)

spider_angle = [0, 90, 180, 270] # in [º]
spider_thickness = 0.060 # in [m]

if spider:
    est_tel.apply_spiders(spider_angle, spider_thickness)

## Sources:
ngs_red = Source(magnitude=5,
             optBand='R4',
             coordinates=[0,0],
             logger=test_logger.logger)

sun_red = ExtendedSource(optBand='R',
                     coordinates=[0, 0],
                     nSubDirs=3,
                     fov=9.269,
                     subDir_margin=4.0,
                     patch_padding=5.0,
                     logger=test_logger.logger)

total_cases = len(atm_cases)
current_case_idx = 0

for atm_name, draws in atm_cases.items():
    atm_idx = int(atm_name.replace('atm', ''))
    if start_atm_idx is not None and atm_idx < start_atm_idx:
        current_case_idx += 1
        continue
        
    draw_name = 'draw1'
    kwargs = draws[draw_name]
    current_case_idx += 1
    
    t_case_start = time.time()
    test_logger.logger.info(f"=== [{current_case_idx}/{total_cases}] Starting CL 0samples Validation for {atm_name} {draw_name} ===")

    atm_file_name = f"ps_val_{atm_name}_{draw_name}.h5"
    atm_file_path = os.path.join(ps_dir, atm_file_name)

    # Define Atmosphere struct just once per draw to guarantee structure maps
    atm = Atmosphere(r0=kwargs['r0'],
                     L0=kwargs['L0'],
                     fractionalR0=kwargs['fractionalR0'],
                     altitude=kwargs['altitude'],
                     windDirection=kwargs['windDirection'],
                     windSpeed=kwargs['windSpeed'],
                     telescope=est_tel,
                     zenith=kwargs['zenith'],
                     logger=test_logger.logger)

    for use_vibrations in [False, True]:
        success = atm.load(atm_file_path)
        if not success:
           test_logger.logger.error(f"Failed to load {atm_file_path}. Skipping.")
           continue
           
        if use_vibrations:
            vibr_label = "Vibr"
            vib_idx = atm_idx if atm_idx <= 5 else 1
            vibration_file = os.path.join(user_home, 'simulations', 'VibrationsSource', f'EST_vibration_{vib_idx}.h5')
            red_vibrations = Vibration(est_tel, vibration_file, test_logger.logger)
        else:
            vibr_label = "noVibr"
            red_vibrations = None

        res_file_name = f"res_0samples_{vibr_label}_{atm_name}_{draw_name}.h5"
        res_file_path = os.path.join(res_dir, res_file_name)

        savepoint = Savepoint(file_path=res_file_path, atm=1, atm_per_dir=1, dm=1, dm_per_dir=1, slopes=1, wfs=1, wfs_frame=1, sci=1, sci_frame=1, only_metrics=1, logger=test_logger.logger)

        asm_params = {'dynamicModel': os.path.join(user_home, 'simulations', 'MirrorModels', 'asm_discrete_model.h5'), 'validActThreshpercentage': 0.5}
        asm = DeformableMirror(telescope=est_tel,
                                nActs=n_subaperture_red+1,
                                altitude=0,
                                typeDM='cartesian',
                                logger=test_logger.logger,
                                **asm_params) # ASM
        dms = [asm]

        red_wfs = CorrelatingShackHartmann(telescope=est_tel,
                                            src=sun_red,
                                            lightRatio=0.9,
                                            nSubap=n_subaperture_red,
                                            plate_scale=0.403,
                                            fieldOfView=9.269,
                                            guardPx=2,
                                            fft_fieldOfView_oversampling=0.5,
                                            use_brightest=9,
                                            unit_in_rad=False,
                                            logger=test_logger.logger)

        red_scicam = ScienceCam(fieldOfView=9.269, 
                                 plate_scale = 0.0167,
                                 samplingTime=est_tel.samplingTime,
                                 telescope=est_tel,
                                 integrationTime=1./scienceFs,
                                 noiseFlag=False,
                                 logger=test_logger.logger)

        scao_light_path_list = []
        # Create red branch (0 delay)
        scao_light_path_list.append(LightPath(test_logger.logger))
        scao_light_path_list[-1].initialize_path(src=sun_red, atm=atm, tel=est_tel, dm=dms[0], wfs=red_wfs, vibration=red_vibrations, sci=None, delay=0)

        # Red branch evaluation point-like source
        scao_light_path_list.append(LightPath(test_logger.logger))
        scao_light_path_list[-1].initialize_path(src=ngs_red, atm=atm, tel=est_tel, dm=dms[0], wfs=None, vibration=red_vibrations, sci=red_scicam, delay=0)

        lightPathTasks = []
        for i in range(len(scao_light_path_list)):
            lightPathTasks.append(delayed(scao_light_path_list[i].propagate)(True))

        im_handler = InteractionMatrixHandler(test_logger.logger)
        im_handler.initialize_im_class(scao_light_path_list)

        if load_modal_basis:
            im_handler.load_modalBasis(load_filename_modalBasis)
        if measure_new_IM:
            im_handler.measure(modal_basis='dh', stroke=im_stroke, nModes=nModes)
            im_handler.save_IM(load_filename_IM)
        else:
            im_handler.load_IM(load_filename_IM)

        controller_kwargs = {'rcond':0.025, 
                            'beta':1e-4,
                            'gain':[0.4],
                            'decay':[0.9999],
                            'ki':[0.0]}

        controller = Controller(telescope=est_tel,
                                interactionMatrix=im_handler,
                                reconstructionMethod='inversion',
                                controllerType='leaky',
                                logger=test_logger.logger,
                                **controller_kwargs)

        test_logger.logger.info(f'Beginning simulation loop for Validation ({vibr_label})')

        # SCAO loop
        for i in range(nIterations):
            if i % 100 == 0:
                est_tel.logger.info(f'Iteration {i+1}')
            atm.update()
            Parallel(n_jobs=1, prefer="threads")(lightPathTasks)
            cmd = controller.computeControlAction(scao_light_path_list)
            for j in range(len(dms)):
                dms[j].updateDMShape(cmd[j])
            
            savepoint.save([atm], i)
            savepoint.save(dms, i)
            savepoint.save(scao_light_path_list, i)

        test_logger.logger.info(f'Simulation ended for {atm_name} {draw_name} ({vibr_label})')

# Force destructor call for the log queue
test_logger = None