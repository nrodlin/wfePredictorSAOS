from datetime import datetime
import time

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

from wfePredictorSAOS.predictor.onlinePredictor import OnlineSlopePredictor

# Logger:

test_logger = LoggingHelper(logging.INFO)

# Simulation settings:

nIterations = 2000

generate_new_atm = True
# Loading files:
load_filename_atm = '/home/oopao/simulations/phase_screens/20260318_1721.h5'

# Saving files
date = datetime.datetime.now().strftime("%Y%m%d_%H%M")

save_filename_atm = '/home/oopao/simulations/phase_screens/' + date

# Define data sharepoint

#sharepoint = Sharepoint(test_logger.logger, port=5572, slopes=1)

# Define the savingpoint
savepoint = Savepoint(file_path='', slopes=1, error=1, logger=test_logger.logger)

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

spider_angle = [0, 90, 180, 270] # in [º]
spider_thickness = 0.060 # in [m]

# est_tel.apply_spiders(spider_angle, spider_thickness)

# Atmosphere:
id = 4

# Draw 3
atm = Atmosphere(r0=0.30,
                 L0=25,
                 fractionalR0=[0.30, 0.18, 0.22, 0.18, 0.12],
                 altitude=[100, 1500, 5000, 10000, 15000],
                 windDirection=[350, 40, 120, 210, 290],
                 windSpeed=[6, 7, 10, 9, 5],
                 telescope=est_tel,
                 zenith=15,
                 logger=test_logger.logger)

if generate_new_atm:
    atm.initializeAtmosphere()
    atm.save(save_filename_atm)
else:
    atm.load(load_filename_atm)

# Sources:
sun = ExtendedSource(optBand = 'R',
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
scao_light_path_list[-1].initialize_path(src=sun, atm=atm, tel=est_tel, dm=None, wfs=shwfs_0, ncpa=None, sci=None, delay=0)

# Create red branch with 2 delay samples
scao_light_path_list.append(LightPath(test_logger.logger))
scao_light_path_list[-1].initialize_path(src=sun, atm=atm, tel=est_tel, dm=None, wfs=shwfs_0, ncpa=None, sci=None, delay=2)

lightPathTasks = []
for i in range(len(scao_light_path_list)):
    lightPathTasks.append(delayed(scao_light_path_list[i].propagate)(True))


### Predictor

past_horizon = 8
n_slopes = scao_light_path_list[0].slopes_1D.shape[0]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

predictor = OnlineSlopePredictor(
    n_slopes=n_slopes,
    model_path='/home/oopao/code/wfePredictorSAOS/best_model_IndepLSTM.pt',
    past_horizon=past_horizon,
    hidden_size=32,
    device=device,
    mean=None,   # sustituir por mean real si la guardaste
    std=None     # sustituir por std real si la guardaste
)

prediction_list = []
ground_truth_list = []
mse_list = []

test_logger.logger.info(f'The Modules initialization took {time.time()-t0} [s]')

test_logger.logger.info('Beginning simulation')


past_horizon = 8
last_measurement = np.zeros((past_horizon, scao_light_path_list[0].slopes_1D.shape[0]))

prediction_list = []
ground_truth_list = []

# SCAO loop
for i in range(nIterations):
    est_tel.logger.info(f'Iteration {i+1}')
    # Update the atmosphere
    atm.update()
    # Propagate the light
    Parallel(n_jobs=2, prefer='threads')(lightPathTasks)
    # Share data with the GUI
    # sharepoint.shareData(scao_light_path_list, i)              
 
    # Save Data
    # savepoint.save(scao_light_path_list, i)

    ## Store in a cyclic buffer
    delayed_slopes = scao_light_path_list[1].slopes_1D.copy()
    predictor.push(delayed_slopes)

    if predictor.ready():
        predict = predictor.predict()
        truth = scao_light_path_list[0].slopes_1D.copy()

        prediction_list.append(predict)
        ground_truth_list.append(truth)

        mse = np.mean((predict - truth) ** 2)
        mse_list.append(mse)

        est_tel.logger.info(f'Prediction MSE = {mse:.6e}')    

if len(mse_list) > 0:
    test_logger.logger.info(f'Mean prediction MSE = {np.mean(mse_list):.6e}')

prediction_array = np.array(prediction_list, dtype=np.float32)
ground_truth_array = np.array(ground_truth_list, dtype=np.float32)

np.save(f'/home/oopao/simulations/results/predictor/prediction_{id}.npy', prediction_array)
np.save(f'/home/oopao/simulations/results/predictor/truth_{id}.npy', ground_truth_array)

test_logger.logger.info('Simulation ended.')

# Force destructor call for the queue of logs

test_logger = None