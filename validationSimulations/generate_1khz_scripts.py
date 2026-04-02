import os
import re

files_to_copy = [
    ('redArmSolarSCAOPredictionCL0samples.py', 'redArmSolarSCAOPredictionCL0samples_1kHz.py'),
    ('redArmSolarSCAOPredictionCL2samples.py', 'redArmSolarSCAOPredictionCL2samples_1kHz.py'),
    ('redArmSolarSCAOPredictionCL2samplesPredictor.py', 'redArmSolarSCAOPredictionCL2samplesPredictor_1kHz.py'),
    ('redArmSolarSCAOPredictionCL2samplesLinearPredictor.py', 'redArmSolarSCAOPredictionCL2samplesLinearPredictor_1kHz.py')
]

for src, dest in files_to_copy:
    with open(src, 'r') as f:
        lines = f.readlines()
        
    out_lines = []
    in_vibr_loop = False
    
    for line in lines:
        # 1. Update suffix in res_dir
        if "res_dir = os.path.join(user_home, 'simulations', 'results', 'cl_" in line:
            line = line.replace("')", "_1kHz')")
            
        # 2. Update sampling_time
        if "sampling_time = 1/2000" in line:
            line = line.replace("1/2000", "1/1000")
            
        # 3. Handle vibration loop initialization
        if "for use_vibrations in [False, True]:" in line:
            in_vibr_loop = True
            out_lines.append("    vibr_label = 'noVibr'\n")
            out_lines.append("    red_vibrations = None\n")
            continue
            
        # 3b. Skip the vibration block safely
        if in_vibr_loop and (line.startswith("        if use_vibrations:") or
                             line.startswith("            vibr_path =") or
                             line.startswith("            red_vibrationsParams =") or
                             line.startswith("            red_vibrations = Vibrations") or
                             line.startswith("            vibr_label = \"Vibr\"") or
                             line.startswith("        else:") or
                             line.startswith("            vibr_label = \"noVibr\"") or
                             line.startswith("            red_vibrations = None")):
            continue
            
        # 4. Handle dynamicModel param (ASM)
        if "asm_params = {" in line and "'dynamicModel'" in line:
            line = re.sub(r"'dynamicModel': os\.path\.join\([^)]+\)", r"'dynamicModel': ''", line)
            
        # 5. Fix the output file name suffix inside the loop
        if "res_file_name =" in line and ".h5" in line:
            line = line.replace("res_0samples_", "res_0samples_1kHz_")
            line = line.replace("res_2samples_", "res_2samples_1kHz_")
            line = line.replace("res_2samplesPredictor_", "res_2samplesPredictor_1kHz_")
            line = line.replace("res_2samplesLinearPredictor_", "res_2samplesLinearPredictor_1kHz_")
            
        # De-indent contents that were inside the loop
        if in_vibr_loop:
            if line.startswith("        "):
                line = line[4:]
            elif line == "    \n" or line == "        \n":
                line = "\n"

        out_lines.append(line)
        
    with open(dest, 'w') as f:
        f.writelines(out_lines)

# Also update run_all_cl.sh
with open('run_all_cl.sh', 'r') as f:
    text = f.read()
text = text.replace('.py', '_1kHz.py')
text = text.replace('CL0samples.py', 'CL0samples_1kHz.py') # Double check
with open('run_all_cl_1kHz.sh', 'w') as f:
    f.write(text)
os.chmod('run_all_cl_1kHz.sh', 0o755)

# Also update analyseResultsCL_Strehl.py
with open('analyseResultsCL_Strehl.py', 'r') as f:
    text = f.read()
text = text.replace("'cl_0samples'", "'cl_0samples_1kHz'")
text = text.replace("'cl_2samples'", "'cl_2samples_1kHz'")
text = text.replace("'cl_2samplesPrediction'", "'cl_2samplesPrediction_1kHz'")
text = text.replace("'cl_2samplesLinearPrediction'", "'cl_2samplesLinearPrediction_1kHz'")
with open('analyseResultsCL_Strehl_1kHz.py', 'w') as f:
    f.write(text)

print("All 1kHz scripts generated successfully!")
