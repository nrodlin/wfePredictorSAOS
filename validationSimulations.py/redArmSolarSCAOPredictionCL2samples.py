from datetime import datetime
import time
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from joblib import Parallel, delayed

import logging
import datetime

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

# Logger:

test_logger = LoggingHelper(logging.INFO)

# Simulation settings:

nIterations = 2000

scienceFs = 56. # Hz

spider = False
generate_new_atm = False
measure_new_IM = False
load_modal_basis = True

nModes = None # [nModesASM]
im_stroke = [5e-7] # in meters

# Loading files:
load_filename_atm = os.path.join(os.path.expanduser("~"), 'simulations/phase_screens/predictor_ps_atm1_draw1.h5')
load_filename_modalBasis = os.path.join(os.path.expanduser("~"), 'simulations/modal_basis/predictor_modalBasis.h5')
load_filename_IM = os.path.join(os.path.expanduser("~"), 'simulations/interaction_matrix/predictor_IM.h5')

# Saving files
date = datetime.datetime.now().strftime("%Y%m%d_%H%M")

save_filename_atm = os.path.join(os.path.expanduser("~"), 'simulations/phase_screens/' + date + '.h5')
save_filename_modalBasis = os.path.join(os.path.expanduser("~"), 'simulations/modal_basis/' + date + '.h5')
save_filename_IM = os.path.join(os.path.expanduser("~"), 'simulations/interaction_matrix/' + date + '.h5')

## Define data sharepoint

sharepoint = Sharepoint(test_logger.logger, port=5573, atm=1, atm_per_dir=0, dm=1, dm_per_dir=1, slopes=1, wfs=1, wfs_frame=0, sci=1, sci_frame=1)

## Define the savingpoint
savepoint = Savepoint(file_path='', atm=1, atm_per_dir=1, dm=1, dm_per_dir=1, slopes=1, wfs=1, wfs_frame=1, sci=1, sci_frame=1, only_metrics=1, logger=test_logger.logger)

## Define EST
t0 = time.time()

diameter = 4.149 # in [m]
obs_diameter = 1.3 # in [m]
sampling_time = 1/2000 # in [s]
n_subaperture_red = 36
resolution = n_subaperture_red * 4 # resolution of the phase screen in [px]
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

if spider:
    est_tel.apply_spiders(spider_angle, spider_thickness)

## Atmosphere:

# Draw 1
atm = Atmosphere(r0=0.06,
                 L0=25,
                 fractionalR0=[0.54, 0.27, 0.10, 0.06, 0.03],
                 altitude=[100, 1500, 5000, 10000, 15000],
                 windDirection=[40, 85, 140, 220, 310],
                 windSpeed=[12, 16, 20, 14, 9],
                 telescope=est_tel,
                 zenith=45,
                 logger=test_logger.logger)

if generate_new_atm:
    atm.initializeAtmosphere()
    atm.save(save_filename_atm)
else:
    atm.load(load_filename_atm)

## Sources:
ngs_red = Source(magnitude = 5,
             optBand = 'R4',
             coordinates=[0,0],
             logger=test_logger.logger)

sun_red = ExtendedSource(optBand = 'R',
                     coordinates=[0, 0],
                     nSubDirs=3,
                     fov=9.269,
                     subDir_margin=4.0,
                     patch_padding=5.0,
                     logger=test_logger.logger)             

## Deformable mirrors:

asm_params = {'dynamicModel': os.path.join(os.path.expanduser("~"), 'simulations/MirrorModels/asm_discrete_model.h5'), 'validActThreshpercentage': 0.5}
# asm_params = {'dynamicModel': '', 'validActThreshpercentage': 0.5}

asm = DeformableMirror(telescope=est_tel,
                        nActs=n_subaperture_red+1,
                        altitude=0,
                        typeDM='cartesian',
                        logger=test_logger.logger,
                        **asm_params) # ASM

dms = [asm]

## Vibration

red_vibration_file = '/home/oopao/simulations/VibrationsSource/EST_vibration_1.h5'

red_vibrations = None#Vibration(est_tel, red_vibration_file, test_logger.logger)

## Wavefront Sensor

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

## Science camera

red_scicam = ScienceCam(fieldOfView=9.269, 
                         plate_scale = 0.0167,
                         samplingTime=est_tel.samplingTime,
                         telescope=est_tel,
                         integrationTime=1./scienceFs,
                         noiseFlag=False,
                         logger=test_logger.logger)

## Build the Light Path

scao_light_path_list = []

# Create red branch
scao_light_path_list.append(LightPath(test_logger.logger))
scao_light_path_list[-1].initialize_path(src=sun_red, atm=atm, tel=est_tel, dm=dms[0], wfs=red_wfs, vibration=red_vibrations, sci=None, delay=2)

# Red branch evaluation point-like source
scao_light_path_list.append(LightPath(test_logger.logger))
scao_light_path_list[-1].initialize_path(src=ngs_red, atm=atm, tel=est_tel, dm=dms[0], wfs=None, vibration=red_vibrations, sci=red_scicam, delay=2)

lightPathTasks = []
for i in range(len(scao_light_path_list)):
    lightPathTasks.append(delayed(scao_light_path_list[i].propagate)(True))

test_logger.logger.info(f'The Modules initialization took {time.time()-t0} [s]')

## Interaction Matrix
t0 = time.time()

im_handler = InteractionMatrixHandler(test_logger.logger)
im_handler.initialize_im_class(scao_light_path_list)

if load_modal_basis:
    im_handler.load_modalBasis(load_filename_modalBasis)
if measure_new_IM:
    im_handler.measure(modal_basis='dh', stroke=im_stroke, nModes=nModes)
    im_handler.save_IM(save_filename_IM)

    if not load_modal_basis:
        im_handler.save_modalBasis(save_filename_modalBasis)
else:
    im_handler.load_IM(load_filename_IM)

test_logger.logger.info(f'The IM creation took {time.time()-t0} [s]')

# Controller class
controller_kwargs = {'rcond':0.025, 
                    'beta':1e-4,
                    'gain':[0.25],
                    'decay':[0.999],
                    'ki':[0.0]}

controller = Controller(telescope=est_tel,
                        interactionMatrix=im_handler,
                        reconstructionMethod='inversion',
                        controllerType='leaky',
                        logger=test_logger.logger,
                        **controller_kwargs)

test_logger.logger.info('Beginning simulation')

# SCAO loop
for i in range(nIterations):
    est_tel.logger.info(f'Iteration {i+1}')
    # Update the atmosphere
    atm.update()
    # Propagate the light
    Parallel(n_jobs=1, prefer="threads")(lightPathTasks)
    # Compute command
    cmd = controller.computeControlAction(scao_light_path_list)
    # Update the DM shape
    for j in range(len(dms)):
        dms[j].updateDMShape(cmd[j])
    # Share data with the GUI
    sharepoint.shareData(scao_light_path_list, i, [atm], dms)              
 
    # Save Data
    savepoint.save([atm], i)
    savepoint.save(dms, i)
    savepoint.save(scao_light_path_list, i)

test_logger.logger.info('Simulation ended.')

# Force destructor call for the qeue of logs

test_logger = None