from datetime import datetime
import time
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from joblib import Parallel, delayed
import logging
import h5py

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

from wfePredictorSAOS.predictor.onlinePredictor import OnlineSlopePredictor
from atmosphereCases import atm_cases

# Logger:
test_logger = LoggingHelper(logging.INFO)

# Simulation settings:
nIterations = 2000
generate_new_atm = True

# Start execution from a specific atmosphere case and draw index
start_atm_idx = 3   # Set to None to start from the beginning
start_draw_idx = 2  # Set to None to start from the first draw

user_home = os.path.expanduser('~')
ps_dir = os.path.join(user_home, 'simulations', 'phase_screens')
res_dir = os.path.join(user_home, 'simulations', 'results')
predictor_res_dir = os.path.join(res_dir, 'predictor_ol')

os.makedirs(ps_dir, exist_ok=True)
os.makedirs(res_dir, exist_ok=True)
os.makedirs(predictor_res_dir, exist_ok=True)

# Define EST
t0 = time.time()

diameter = 4.149 # in [m]
obs_diameter = 1.3 # in [m]
sampling_time = 1/2000 # in [s]
n_subaperture = 36
resolution = n_subaperture * 4 # resolution of the phase screen in [px]
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
# est_tel.apply_spiders(spider_angle, spider_thickness)

total_cases = len(atm_cases) * 3
current_case_idx = 0

for atm_name, draws in atm_cases.items():
    atm_idx = int(atm_name.replace('atm', ''))
    if start_atm_idx is not None and atm_idx < start_atm_idx:
        current_case_idx += len(draws)
        continue
        
    for draw_name, kwargs in draws.items():
        current_case_idx += 1
        
        draw_idx = int(draw_name.replace('draw', ''))
        if start_atm_idx is not None and start_draw_idx is not None:
            if atm_idx == start_atm_idx and draw_idx < start_draw_idx:
                continue

        t_case_start = time.time()
        test_logger.logger.info(f"=== [{current_case_idx}/{total_cases}] Starting OL Validation for {atm_name} {draw_name} (LOS r0: {kwargs['los_r0']:.2f} m) ===")
        
        atm_file_name = f"ps_val_{atm_name}_{draw_name}.h5"
        atm_file_path = os.path.join(ps_dir, atm_file_name)
        
        # Define Atmosphere struct just once per draw to guarantee the initialization structure maps
        atm = Atmosphere(r0=kwargs['r0'],
                         L0=kwargs['L0'],
                         fractionalR0=kwargs['fractionalR0'],
                         altitude=kwargs['altitude'],
                         windDirection=kwargs['windDirection'],
                         windSpeed=kwargs['windSpeed'],
                         telescope=est_tel,
                         zenith=kwargs['zenith'],
                         logger=test_logger.logger)
                         
        if generate_new_atm:
            # Generate cases on runtime avoiding load failures if missing
            atm.initializeAtmosphere()
            atm.save(atm_file_path)
            test_logger.logger.info(f"Generated new atmosphere -> {atm_file_path}")
            
        for use_vibrations in [False, True]:
            # By loading within this loop we guarantee that atmosphere displ_buffers are reset,
            # providing perfectly equivalent initial conditions for the sequence with/without vibrations
            success = atm.load(atm_file_path)
            if not success:
               test_logger.logger.error(f"Failed to load {atm_file_path}. Skipping.")
               continue
            
            if use_vibrations:
                vibr_label = "Vibr"
                atm_num = int(atm_name.replace('atm', ''))
                vib_idx = atm_num if atm_num <= 5 else 1
                vibration_file = f'{user_home}/simulations/VibrationsSource/EST_vibration_{vib_idx}.h5'
                vibrations = Vibration(est_tel, vibration_file, test_logger.logger)
            else:
                vibr_label = "noVibr"
                vibrations = None

            res_file_name = f"res_ol_{vibr_label}_{atm_name}_{draw_name}.h5"
            res_file_path = os.path.join(res_dir, res_file_name)
            
            # Define the savingpoint
            savepoint = Savepoint(file_path=res_file_path, slopes=1, error=1, logger=test_logger.logger)
            
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

            ### Predictor
            past_horizon = 8
            n_slopes = scao_light_path_list[0].slopes_1D.shape[0]

            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            predictor = OnlineSlopePredictor(
                n_slopes=n_slopes,
                model_path=os.path.join(user_home, 'code', 'best_model_IndepLSTM.pt'),
                past_horizon=past_horizon,
                hidden_size=32,
                device=device,
                mean=None,  
                std=None    
            )

            test_logger.logger.info(f'Beginning simulation loop for Validation ({vibr_label})')
            
            prediction_list = []
            ground_truth_list = []
            mse_list = []

            # SCAO loop
            for i in range(nIterations):
                if i % 100 == 0:
                    est_tel.logger.info(f'Iteration {i+1}')
                # Update the atmosphere
                atm.update()
                # Propagate the light
                Parallel(n_jobs=2, prefer='threads')(lightPathTasks)
                
                # Save Data in H5 res_ol_noVibr_atmX_drawY.h5 or Vibr
                savepoint.save(scao_light_path_list, i)

                ## Predictor Data accumulation
                delayed_slopes = scao_light_path_list[1].slopes_1D.copy()
                predictor.push(delayed_slopes)

                if predictor.ready():
                    predict = predictor.predict()
                    truth = scao_light_path_list[0].slopes_1D.copy()

                    prediction_list.append(predict)
                    ground_truth_list.append(truth)

                    mse = np.mean((predict - truth) ** 2)
                    mse_list.append(mse)

                    if i % 100 == 0:
                         est_tel.logger.info(f'Prediction MSE = {mse:.6e}')    

            if len(mse_list) > 0:
                test_logger.logger.info(f'Mean prediction MSE for {atm_name} {draw_name} ({vibr_label}) = {np.mean(mse_list):.6e}')

            prediction_array = np.array(prediction_list, dtype=np.float32)
            ground_truth_array = np.array(ground_truth_list, dtype=np.float32)

            np.save(os.path.join(predictor_res_dir, f'prediction_val_{vibr_label}_{atm_name}_{draw_name}.npy'), prediction_array)
            np.save(os.path.join(predictor_res_dir, f'truth_val_{vibr_label}_{atm_name}_{draw_name}.npy'), ground_truth_array)

            test_logger.logger.info(f'Simulation ended for {atm_name} {draw_name} ({vibr_label})')

        t_case_end = time.time()
        test_logger.logger.info(f'=== Completed OL Validation for {atm_name} {draw_name} in {t_case_end - t_case_start:.2f} [s] ===')

# Force destructor call for the queue of logs
test_logger = None