# wfePredictorSAOS

Temporal prediction of the wavefront-sensor slopes for the EST red-arm SCAO loop (2 kHz, Correlating Shack-Hartmann on the sun,
ASM 36x36 / M7 50x50), simulated with [SAOS](https://github.com/nrodlin/SAOS). The predictor works on the pseudo-open-loop (POL) slopes
and feeds the POL control law.

## Layout

```
wfePredictorSAOS/predictor/   online predictors (push / ready / predict) and the POL reconstruction
    incrementalLSTM.py          LSTM v2: incremental prediction, per-window normalisation, multi-horizon
    incrementARPredictor.py     increment AR baseline (+ fit CLI)
    onlineLinearPredictor.py    linear extrapolation
    onlinePredictor.py          LSTM v1 (raw slopes, reference)
    stencilPredictor.py         spatio-temporal 3x3 stencil, global or adaptive (RLS)
    pseudoOpenLoop.py           POL slopes with the DM dynamics replica
wfePredictorSAOS/training/    trainIncrementalLSTM.py (band-limited loss)
wfePredictorSAOS/weights/     trained weights (.pt, .npz)
simulations/                  common.py (configuration shared by every script)
    prepare_assets.py           modal bases and interaction matrices
    generateTrainingData.py     dataset v2: POL slopes of a closed loop + true open-loop slopes, random conditions
    polPredictionErrors.py      3 light paths on one WFS: POL error and prediction errors under the same atmosphere
    closedLoopComparison.py     paired closed loop: integrator and POL with each predictor
    jobs/                       Slurm jobs (durum)
```

## Workflow

1. `prepare_assets.py` (once): modal bases and IMs in `<base_dir>/modal_basis`, `<base_dir>/interaction_matrix`.
2. Dataset: `sbatch jobs/job_dataset_v2.job` (train 36x36) and the val / test (36x36, 50x50) arrays given in its header.
3. Training: `sbatch jobs/job_train_v2.job` (LSTM v2 with band and MSE losses, AR baselines).
4. Evaluation: `sbatch jobs/job_evaluation_v2.job` (3 light paths and closed-loop comparison, 36x36 and 50x50).

Predictor names (simulations/common.py `make_predictor`): `zoh`, `linear`, `lstm_v1`, `lstm_v2[:tag]`, `incAR:<weights file>`, `stencil_rls`.
