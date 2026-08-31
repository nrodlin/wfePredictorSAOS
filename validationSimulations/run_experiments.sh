#!/bin/bash
# ==============================================================================
# Master Runner for 2kHz AO Predictor Simulations (36x36 & 50x50)
# ==============================================================================

PYTHON_CMD="/home/nlinares/.pyenv/versions/3.13.9t/envs/saos/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SENSORS=("36" "50")
N_ITERATIONS=2000

echo "=============================================================================="
echo " Starting AO Predictor Simulation Campaign (2 kHz)"
echo " Sensors: ${SENSORS[*]}"
echo " Iterations per simulation: $N_ITERATIONS"
echo " Start Time: $(date)"
echo "=============================================================================="

for SENSOR in "${SENSORS[@]}"; do
    echo ""
    echo "##############################################################################"
    echo " RUNNING EXPERIMENTS FOR SENSOR ${SENSOR}x${SENSOR}"
    echo "##############################################################################"

    # 1. Open Loop Simulation
    echo ""
    echo ">>> [1/4] Running Open Loop (OL) Validation [${SENSOR}x${SENSOR}]..."
    t_start=$(date +%s)
    $PYTHON_CMD redArmSolarSCAO_01_OL.py --sensor "$SENSOR" --n_iterations "$N_ITERATIONS"
    t_end=$(date +%s)
    echo ">>> [1/4] Finished OL [${SENSOR}x${SENSOR}] in $((t_end - t_start)) s."

    # 2. Closed Loop Baseline Simulation
    echo ""
    echo ">>> [2/4] Running Closed Loop Baseline (delay=2) [${SENSOR}x${SENSOR}]..."
    t_start=$(date +%s)
    $PYTHON_CMD redArmSolarSCAO_02_CL_baseline.py --sensor "$SENSOR" --delay 2 --n_iterations "$N_ITERATIONS"
    t_end=$(date +%s)
    echo ">>> [2/4] Finished CL Baseline [${SENSOR}x${SENSOR}] in $((t_end - t_start)) s."

    # 3. Closed Loop POL Simulations (Linear & LSTM)
    echo ""
    echo ">>> [3a/4] Running Closed Loop POL (Linear Predictor) [${SENSOR}x${SENSOR}]..."
    t_start=$(date +%s)
    $PYTHON_CMD redArmSolarSCAO_03_CL_POL.py --sensor "$SENSOR" --predictor linear --n_iterations "$N_ITERATIONS"
    t_end=$(date +%s)
    echo ">>> [3a/4] Finished CL POL Linear [${SENSOR}x${SENSOR}] in $((t_end - t_start)) s."

    echo ""
    echo ">>> [3b/4] Running Closed Loop POL (LSTM Predictor) [${SENSOR}x${SENSOR}]..."
    t_start=$(date +%s)
    $PYTHON_CMD redArmSolarSCAO_03_CL_POL.py --sensor "$SENSOR" --predictor lstm --n_iterations "$N_ITERATIONS"
    t_end=$(date +%s)
    echo ">>> [3b/4] Finished CL POL LSTM [${SENSOR}x${SENSOR}] in $((t_end - t_start)) s."

    # 4. Closed Loop Sin Cerrar Simulation
    echo ""
    echo ">>> [4/4] Running Closed Loop 'Sin Cerrar' Monitoring [${SENSOR}x${SENSOR}]..."
    t_start=$(date +%s)
    $PYTHON_CMD redArmSolarSCAO_04_CL_sin_cerrar.py --sensor "$SENSOR" --n_iterations "$N_ITERATIONS"
    t_end=$(date +%s)
    echo ">>> [4/4] Finished CL Sin Cerrar [${SENSOR}x${SENSOR}] in $((t_end - t_start)) s."
done

echo ""
echo "=============================================================================="
echo " ALL SIMULATIONS FINISHED! Running Analysis Report..."
echo " End Time: $(date)"
echo "=============================================================================="

$PYTHON_CMD analyse_comparison.py
