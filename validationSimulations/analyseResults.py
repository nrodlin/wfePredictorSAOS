import numpy as np
import matplotlib.pyplot as plt

## Comparar mejora al hacer la predicción vs medida 2 instantes tarde
atmosphere = 4
nDraws = 3
path = '/home/oopao/simulations/results/predictor/'
delay = 2

error_prediction_draw = []
error_delay_draw      = []

for i in range(nDraws):
    prediction_file = f'prediction_atm{atmosphere}_draw{i+1}.npy'
    truth_file      = f'truth_atm{atmosphere}_draw{i+1}.npy'

    prediction = np.load(path + prediction_file)
    truth      = np.load(path + truth_file)

    delayed_truth = truth[delay:,:]

    error_prediction = []
    error_delay      = []
    
    for j in range(prediction.shape[0]):
        # Compare the erorr of the prediction
        error_prediction.append(np.sqrt(np.mean((prediction[j,:]-truth[j, :])**2)))
    for j in range(delayed_truth.shape[0]):
        # Compute the error assuming a delayed measurement
        error_delay.append(np.sqrt(np.mean((delayed_truth[j,:]-truth[j, :])**2))) # delayed_truth is 2 samples ahead --> case without delay w.r.t truth
    
    error_prediction_draw.append(error_prediction.copy())
    error_delay_draw.append(error_delay.copy())

# ---- Subplots ----
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

handles = []
labels = []

# --- Prediction error ---
for idx_fila, fila in enumerate(error_prediction_draw):
    line, = axs[0].plot(fila)  # sin label aquí
    handles.append(line)
    labels.append(f"Draw {idx_fila+1}")

axs[0].set_ylabel("Prediction Error [px]")
axs[0].set_title("Prediction Error")
axs[0].grid(True)

# --- Delay error ---
for idx_fila, fila in enumerate(error_delay_draw):
    axs[1].plot(fila)  # no label

axs[1].set_xlabel("Sample")
axs[1].set_ylabel("Delay Error [px]")
axs[1].set_title("Delay Error")
axs[1].grid(True)

# --- Global title ---
fig.suptitle(f"Atmosphere {atmosphere}")

# --- Leyenda única ---
fig.legend(handles, labels, loc="upper right", ncol=2)

plt.tight_layout()
plt.show()
# Print the metrics

print(f'Avrg. prediction error: {np.mean(error_prediction_draw, axis=1)} [px], Std.: {np.std(error_prediction_draw, axis=1)} [px]')
print(f'Avrg. delay error: {np.mean(error_delay_draw, axis=1)} [px], Std.: {np.std(error_delay_draw, axis=1)} [px]')

print('END')



