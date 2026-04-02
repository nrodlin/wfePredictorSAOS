import h5py
import numpy as np
import os
import glob

# Data folders saved in CL scripts
folders = ['cl_0samples', 'cl_2samples', 'cl_2samplesPrediction', 'cl_2samplesLinearPrediction']
base_path = os.path.expanduser('~/simulations/results')

print("===================================================================================================================")
print(f"{'Configuration (CL)':<30} | {'Atmosphere':<12} | {'Draw':<8} | {'Vibration':<12} | {'Avg. Strehl':<15} | {'Std. Strehl':<15}")
print("-------------------------------------------------------------------------------------------------------------------")

results = []

for folder in folders:
    folder_path = os.path.join(base_path, folder)
    if not os.path.exists(folder_path):
        continue
    
    # Search for generated files
    h5_files = glob.glob(os.path.join(folder_path, '*.h5'))
    for file_path in h5_files:
        filename = os.path.basename(file_path)
        
        # Parses names like: res_0samples_noVibr_atm1_draw1.h5
        parts = filename.replace('.h5', '').split('_')
        if len(parts) >= 5:
            vibr_label = parts[2]
            atm_label = parts[3]
            draw_label = parts[4]
        else:
            continue
            
        try:
            with h5py.File(file_path, 'r') as f:
                path_strehl = 'LightPath_1/sci_frame_longExp/strehl'
                if path_strehl in f:
                    strehl_array = f[path_strehl][()]
                    
                    if len(strehl_array) > 2:
                        # Discard the first 2 elements to remove the transient regime
                        avg_strehl = np.mean(strehl_array[2:])
                        std_strehl = np.std(strehl_array[2:])
                        results.append({
                            'case': folder, 
                            'atm': atm_label, 
                            'draw': draw_label, 
                            'vibr': vibr_label, 
                            'strehl': avg_strehl,
                            'std': std_strehl
                        })
                    else:
                        print(f"[{filename}] Warning: Array has {len(strehl_array)} elements, >2 required.")
                else:
                    print(f"[{filename}] Warning: Path {path_strehl} not found in H5.")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Sort the results by Atmosphere -> Draw -> Vibration -> Case
results.sort(key=lambda x: (int(x['atm'].replace('atm', '')), int(x['draw'].replace('draw', '')), x['vibr'], x['case']))

# Print the table
for r in results:
    strehl_str = f"{r['strehl']:.7f}"
    std_str = f"{r['std']:.7f}"
    print(f"{r['case']:<30} | {r['atm']:<12} | {r['draw']:<8} | {r['vibr']:<12} | {strehl_str:<15} | {std_str:<15}")

print("===================================================================================================================")

# Print summary of global averages by configuration
print("\n--- GLOBAL AVERAGE SUMMARY BY SIMULATION TYPE (OVER ALL ATMOSPHERES) ---")
for folder in folders:
    filtered_strehl = [r['strehl'] for r in results if r['case'] == folder]
    filtered_std = [r['std'] for r in results if r['case'] == folder]
    if filtered_strehl:
        global_avg = np.mean(filtered_strehl)
        global_std_mean = np.mean(filtered_std)
        print(f"-> {folder:<30}: Global Average Strehl {global_avg: .7f} ± {global_std_mean: .7f}")
