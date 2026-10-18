"""
Common configuration of the simulations of the predictor (EST red arm SCAO at 2 kHz by default, Correlating SH on the sun):
directories, telescope, optics (36x36 ASM / 50x50 M7), atmospheres, control (IM, Tikhonov reconstructor, POL replica) and the
factory of the online predictors used by polPredictionErrors.py and closedLoopComparison.py.
"""
import os
import numpy as np
import torch

from SAOS.Source import Source
from SAOS.ExtendedSource import ExtendedSource
from SAOS.Telescope import Telescope
from SAOS.Atmosphere import Atmosphere
from SAOS.DeformableMirror import DeformableMirror
from SAOS.CorrelatingShackHartmann import CorrelatingShackHartmann
from SAOS.InteractionMatrixHandler import InteractionMatrixHandler
from SAOS.Controller import Controller

from wfePredictorSAOS.predictor.pseudoOpenLoop import PseudoOpenLoop

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'wfePredictorSAOS', 'weights')
FS = 2000.0

# Campaign: results under <base_dir>/results/wfePredictor/<CAMPAIGN>, and the IMs and weights of a campaign other than v2 carry its
# name (v2: WFS with fully illuminated subapertures only; v3: SAOS WFS with partially illuminated subapertures, develop 3534fa7)
CAMPAIGN = os.environ.get('WFEPRED_CAMPAIGN', 'v2')
ASSET_TAG = '' if CAMPAIGN == 'v2' else f'_{CAMPAIGN}'


def results_root(base_dir):
    return os.path.join(base_dir, 'results', 'wfePredictor', CAMPAIGN)


def im_file(base_dir, sensor):
    return os.path.join(base_dir, 'interaction_matrix', f'predictor_{sensor}x{sensor}_IM{ASSET_TAG}.h5')

# Noisy WFS (--wfs_noise): detector of the CorrelatingShackHartmann defaults (pco.dimax), dark not subtracted (darkCalibration 0: the
# SAOS dark subtraction wraps negative values in the uint16 cast); the photon level is set with lightRatio
NOISE_KWARGS = dict(darkCalibration=0)

# Profile 1 of the mcaobenchcodes studies (dmRotation SCAO/MCAO6, imageQuality), draw 1
MCAO_PROFILE1 = {'r0': 0.09, 'L0': 25, 'zenith': 15, 'fractionalR0': [0.53, 0.37, 0.05, 0.03, 0.02], 'altitude': [100, 1500, 5000, 10000, 15000],
                 'windSpeed': [6.7, 8.5, 12.7, 23.4, 8.3], 'windDirection': [312, 47, 198, 123, 276]}


def get_base_dir(custom_base_dir=None):
    if custom_base_dir:
        return custom_base_dir
    user = os.environ.get('USER', 'nlinares')
    if os.path.exists('/net/durum/scratch'):
        scratch_dir = f'/net/durum/scratch/{user}/simulations'
        os.makedirs(scratch_dir, exist_ok=True)
        return scratch_dir
    return os.path.join(os.path.expanduser('~'), 'simulations')


def _first_existing(name, base_dir):
    user = os.environ.get('USER', 'nlinares')
    for d in (os.path.join(base_dir, name), f'/net/durum/scratch/{user}/simulations/{name}', f'/mnt/nas-mcao/{name}'):
        if os.path.exists(d):
            return d
    return os.path.join(os.path.expanduser('~'), 'simulations', name)


def get_asset_dirs(base_dir):
    """Directories of the DM dynamic models and of the vibration files."""
    return _first_existing('MirrorModels', base_dir), _first_existing('VibrationsSource', base_dir)


def make_telescope(logger, fs=FS):
    return Telescope(diameter=4.149, resolution=200, centralObstruction=1.3 / 4.149, samplingTime=1 / fs, fov=60.0, logger=logger)


def resampled_asset(path, fs, kind):
    """
    Copy of a 2 kHz asset at the loop rate fs (<name>_fs<fs>.h5 next to it, created the first time); the same path at 2 kHz.
      kind 'dm'        : discrete state space of the DM, converted with the zero-order-hold equivalence (matrix log of the
                         augmented [[A, B], [0, 1]] at Ts, matrix exponential at 1/fs), so the continuous response is kept
      kind 'vibration' : vibration sequence sampled at t = n/fs (linear interpolation of the 2 kHz sequence)
    """
    if fs == FS:
        return path
    out = path.replace('.h5', f'_fs{fs:g}.h5')
    if os.path.exists(out):
        return out
    import h5py
    with h5py.File(path, 'r') as f:
        ts = float(f['A' if kind == 'dm' else 'x_axis'].attrs['Ts'])
        data = {k: f[k]['data'][()] for k in f.keys()}
    if kind == 'dm':
        from scipy.linalg import logm, expm
        A, B = np.atleast_2d(data['A']), np.ravel(data['B'])
        n = A.shape[0]
        M = np.eye(n + 1)
        M[:n, :n], M[:n, n] = A, B
        L = logm(M)
        if np.abs(L.imag).max() > 1e-9:
            raise ValueError(f'resampled_asset - {path}: no real continuous equivalent')
        Md = expm(L.real * (1 / fs) / ts)
        data['A'], data['B'] = Md[:n, :n], Md[:n, n]
        attrs_key = 'A'
    else:
        t_old = np.arange(len(data['x_axis'])) * ts
        t_new = np.arange(0, t_old[-1], 1 / fs)
        data = {k: np.interp(t_new, t_old, v) for k, v in data.items()}
        attrs_key = 'x_axis'
    with h5py.File(out, 'w') as f:
        for k, v in data.items():
            f.create_group(k).create_dataset('data', data=v)
        f[attrs_key].attrs['Ts'] = 1 / fs
        f.attrs['source'] = path
    return out


def build_optics(sensor, tel, mirror_models_dir, logger, fs=FS, light_ratio=None):
    """Sun (WFS source), on-axis NGS (science), DM and a WFS factory for the 36x36 (ASM) or 50x50 (M7) configuration.
    Noiseless WFS, or noisy with light_ratio (fraction of the photons that reach the WFS) if given."""
    if sensor == 36:
        sun = ExtendedSource(optBand='R', coordinates=[0, 0], nSubDirs=3, fov=9.269, subDir_margin=4.0, patch_padding=5.0, logger=logger)
        ngs = Source(magnitude=5, optBand='R4', coordinates=[0, 0], logger=logger)
        dm = DeformableMirror(telescope=tel, nActs=37, altitude=0, typeDM='radial', logger=logger,
                              dynamicModel=resampled_asset(os.path.join(mirror_models_dir, 'asm_discrete_model.h5'), fs, 'dm'), validActThreshpercentage=0.7533)
        wfs_kw = dict(plate_scale=0.403, fieldOfView=9.269)
    elif sensor == 50:
        sun = ExtendedSource(optBand='V', coordinates=[0, 0], nSubDirs=3, fov=9.975, subDir_margin=4.0, patch_padding=5.0, logger=logger)
        ngs = Source(magnitude=5, optBand='V0', coordinates=[0, 0], logger=logger)
        dm = DeformableMirror(telescope=tel, nActs=51, altitude=0, typeDM='cartesian', logger=logger,
                              dynamicModel=resampled_asset(os.path.join(mirror_models_dir, 'm7_discrete_model.h5'), fs, 'dm'), validActThreshpercentage=0.7533)
        wfs_kw = dict(plate_scale=0.475, fieldOfView=9.975)
    else:
        raise ValueError(f'Unknown sensor {sensor}')

    def make_wfs():
        noise = {} if light_ratio is None else NOISE_KWARGS
        return CorrelatingShackHartmann(telescope=tel, src=sun, lightRatio=0.9 if light_ratio is None else light_ratio, nSubap=sensor, guardPx=2,
                                        fft_fieldOfView_oversampling=0.5, use_brightest=9, unit_in_rad=False, noiseFlag=light_ratio is not None,
                                        logger=logger, **wfs_kw, **noise)
    return sun, ngs, dm, make_wfs


def load_atmosphere(tel, kwargs, path, seed, logger):
    """Atmosphere with the parameters kwargs and its own phase-screen file, generated with the seed the first time and reused afterwards."""
    atm = Atmosphere(r0=kwargs['r0'], L0=kwargs['L0'], fractionalR0=kwargs['fractionalR0'], altitude=kwargs['altitude'],
                     windDirection=kwargs['windDirection'], windSpeed=kwargs['windSpeed'], telescope=tel, zenith=kwargs['zenith'], logger=logger)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        atm.initializeAtmosphere(randomState=seed)
        atm.save(path)
        logger.info(f'Generated phase screens {path} (seed {seed})')
    if not atm.load(path):
        raise RuntimeError(f'Could not load {path}')
    return atm


class PolControl:
    """
    IM, Tikhonov reconstructor, modal basis and POL replica of the DM dynamics of one sensing light path. Control laws (modal command c):
      integrator : c = g*(-R s_res) + decay*c
      pol        : c = g*(-R s_target - c) + decay*c, with s_target the POL slopes or their prediction
    """

    def __init__(self, base_dir, sensor, lights_paths, dm, delay, tel, logger, beta=5e-4, decay=0.999, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.decay = decay
        im_handler = InteractionMatrixHandler(logger)
        im_handler.initialize_im_class(lights_paths)
        im_handler.load_modalBasis(os.path.join(base_dir, 'modal_basis', f'predictor_{sensor}x{sensor}_modalBasis.h5'))
        im_handler.load_IM(im_file(base_dir, sensor))
        IM = im_handler.interaction_matrix_warehouse[0][0]['IM']
        controller = Controller(telescope=tel, interactionMatrix=im_handler, reconstructionMethod='tikhonov', controllerType='leaky', logger=logger,
                                rcond=0.025, beta=beta, gain=[0.0], decay=[decay], ki=[0.0], nModes=[IM.shape[1]])
        self.n_modes = controller.reconstructor[0].shape[0]
        offset = controller.discarded_modes[0]
        self.R = controller.reconstructor[0].to(self.device)
        self.M = controller.modal_basis[0][:, offset:offset + self.n_modes].to(self.device)
        self.pol = PseudoOpenLoop(IM, dm, delay, self.n_modes, device=self.device)
        self.dm = dm
        self.cmd = torch.zeros((self.n_modes, 1), dtype=torch.float64, device=self.device)

    def tensor(self, slopes):
        return torch.as_tensor(slopes, dtype=torch.float64, device=self.device).reshape(-1, 1)

    def pol_slopes(self, s_res):
        """POL slopes of the instant measured by s_res (tensor)."""
        return self.pol.reconstruct(s_res)

    def step_integrator(self, s_res, gain):
        self.cmd = gain * (-(self.R @ s_res)) + self.decay * self.cmd
        self._apply()

    def step_pol(self, s_target, gain):
        self.cmd = gain * (-(self.R @ s_target) - self.cmd) + self.decay * self.cmd
        self._apply()

    def _apply(self):
        self.dm.updateDMShape(self.M @ self.cmd)
        self.pol.update(self.cmd)


def make_predictor(name, n_slopes, horizon, valid_subapertures=None, linear_past=4, device=None):
    """
    Online predictors of the open-loop slopes h samples ahead, all with push(slopes) / ready() / predict():
      zoh            None (the caller uses the last sample)
      linear         line fitted to the last linear_past samples and extrapolated (as in the first campaign)
      lstm_v1        first LSTM (raw slopes, absolute target, horizon 2 only)
      lstm_v2[:tag]  incremental LSTM with per-window normalisation (weights/lstm_v2[_tag].pt), multi-horizon
      incAR:<file>   increment AR predictor, weights/<file>.npz
      stencil_rls    adaptive 3x3 stencil (RLS), needs the valid subapertures
    """
    if name == 'zoh':
        return None
    if name == 'linear':
        from wfePredictorSAOS.predictor.onlineLinearPredictor import OnlineLinearSlopePredictor
        return OnlineLinearSlopePredictor(n_slopes=n_slopes, past_horizon=linear_past, steps_ahead=horizon)
    if name == 'lstm_v1':
        from wfePredictorSAOS.predictor.onlinePredictor import OnlineSlopePredictor
        if horizon != 2:
            raise ValueError('lstm_v1 is trained for horizon 2 only')
        return OnlineSlopePredictor(n_slopes=n_slopes, model_path=os.path.join(WEIGHTS_DIR, 'lstm_v1_IndepLSTM.pt'), past_horizon=24, hidden_size=16, n_axis=1)
    if name.startswith('lstm_v2'):
        from wfePredictorSAOS.predictor.incrementalLSTM import OnlineIncrementalLSTMPredictor
        return OnlineIncrementalLSTMPredictor(n_slopes=n_slopes, model_path=os.path.join(WEIGHTS_DIR, name.replace(':', '_') + '.pt'), horizon=horizon, device=device)
    if name.startswith('incAR:'):
        from wfePredictorSAOS.predictor.incrementARPredictor import IncrementARPredictor
        return IncrementARPredictor(n_slopes=n_slopes, weights_file=os.path.join(WEIGHTS_DIR, name.split(':', 1)[1] + '.npz'), horizon=horizon)
    if name == 'stencil_rls':
        from wfePredictorSAOS.predictor.stencilPredictor import StencilPredictor
        return StencilPredictor(valid_subapertures, K=3, past_horizon=8, horizon=horizon, mode='rls', lam=0.995, init='global',
                                weights_file=os.path.join(WEIGHTS_DIR, 'stencil_global_K3_p8.npz'), device=device)
    raise ValueError(f'Unknown predictor {name}')


def opd_rms_nm(opd, pupil):
    return float(np.std(opd[pupil > 0]) * 1e9)
