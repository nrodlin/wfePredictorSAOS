#!/bin/bash

# Activate the virtual environment or use the absolute python path if needed
PYTHON_CMD="/home/nlinares/_1kHz.pyenv/versions/3.13.9t/envs/saos/bin/python"

echo "================================================"
echo "Starting Closed-Loop (CL) Simulations"
echo "Start date & time: $(date)"
echo "================================================"

echo ""
echo "[1/4] Running: redArmSolarSCAOPredictionCL0samples_1kHz.py..."
start=$(date +%s)
$PYTHON_CMD redArmSolarSCAOPredictionCL0samples_1kHz.py
end=$(date +%s)
echo "[1/4] Finished: redArmSolarSCAOPredictionCL0samples_1kHz.py"
echo ">> Time elapsed: $((end-start)) seconds (~$(((end-start)/60)) minutes) <<"

echo ""
echo "[2/4] Running: redArmSolarSCAOPredictionCL2samples_1kHz.py..."
start=$(date +%s)
$PYTHON_CMD redArmSolarSCAOPredictionCL2samples_1kHz.py
end=$(date +%s)
echo "[2/4] Finished: redArmSolarSCAOPredictionCL2samples_1kHz.py"
echo ">> Time elapsed: $((end-start)) seconds (~$(((end-start)/60)) minutes) <<"

echo ""
echo "[3/4] Running: redArmSolarSCAOPredictionCL2samplesPredictor_1kHz.py..."
start=$(date +%s)
$PYTHON_CMD redArmSolarSCAOPredictionCL2samplesPredictor_1kHz.py
end=$(date +%s)
echo "[3/4] Finished: redArmSolarSCAOPredictionCL2samplesPredictor_1kHz.py"
echo ">> Time elapsed: $((end-start)) seconds (~$(((end-start)/60)) minutes) <<"

echo ""
echo "[4/4] Running: redArmSolarSCAOPredictionCL2samplesLinearPredictor_1kHz.py..."
start=$(date +%s)
$PYTHON_CMD redArmSolarSCAOPredictionCL2samplesLinearPredictor_1kHz.py
end=$(date +%s)
echo "[4/4] Finished: redArmSolarSCAOPredictionCL2samplesLinearPredictor_1kHz.py"
echo ">> Time elapsed: $((end-start)) seconds (~$(((end-start)/60)) minutes) <<"

echo ""
echo "================================================"
echo "ALL SIMULATIONS HAVE FINISHED."
echo "End date & time: $(date)"
echo "================================================"
