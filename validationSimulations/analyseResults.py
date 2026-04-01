import numpy as np
import os
import glob

path = os.path.expanduser('~/simulations/results/predictor_ol/')
delay = 2

# Search for all truth files to identify available cases
truth_files = glob.glob(os.path.join(path, 'truth_val_*.npy'))

results = []

for truth_path in truth_files:
    filename = os.path.basename(truth_path)
    # filename format: truth_val_{Vibr|noVibr}_{atm}_{draw}.npy
    parts = filename.replace('.npy', '').split('_')
    if len(parts) != 5:
        continue
        
    vibr_label = parts[2]
    atm_label = parts[3]
    draw_label = parts[4]
    
    pred_filename = filename.replace('truth', 'prediction')
    pred_path = os.path.join(path, pred_filename)
    
    if not os.path.exists(pred_path):
        continue
        
    truth = np.load(truth_path)
    prediction = np.load(pred_path)
    
    delayed_truth = truth[delay:,:]
    
    error_prediction = []
    error_delay      = []
    
    for j in range(prediction.shape[0]):
        error_prediction.append(np.sqrt(np.mean((prediction[j,:]-truth[j, :])**2)))
    for j in range(delayed_truth.shape[0]):
        error_delay.append(np.sqrt(np.mean((delayed_truth[j,:]-truth[j, :])**2)))
        
    avg_pred = np.mean(error_prediction)
    std_pred = np.std(error_prediction)
    
    avg_delay = np.mean(error_delay)
    std_delay = np.std(error_delay)
    
    results.append({
        'atm': atm_label,
        'draw': draw_label,
        'vibr': vibr_label,
        'avg_pred': avg_pred,
        'std_pred': std_pred,
        'avg_delay': avg_delay,
        'std_delay': std_delay
    })

# Sort results
results.sort(key=lambda x: (x['vibr'], int(x['atm'].replace('atm', '')), int(x['draw'].replace('draw', ''))))

print("="*105)
print(f"{'Atmosphere':<12} {'Draw':<8} {'Vibration':<10} | {'Pred Error (px)':<20} | {'Delay Error (px)':<20} | {'Improvement':<12}")
print("-" * 105)

for r in results:
    pred_str = f"{r['avg_pred']:.4f} ± {r['std_pred']:.4f}"
    delay_str = f"{r['avg_delay']:.4f} ± {r['std_delay']:.4f}"
    impr = (r['avg_delay'] - r['avg_pred']) / r['avg_delay'] * 100
    impr_str = f"{impr:+.1f}%"
    print(f"{r['atm']:<12} {r['draw']:<8} {r['vibr']:<10} | {pred_str:<20} | {delay_str:<20} | {impr_str:<12}")

print("="*105)
