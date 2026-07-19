import os
import pandas as pd
import numpy as np
import scipy.signal as signal

# Define plant identifiers
cf_plants = [
    '2023-08-11cf', '2023-08-24cf', '2023-08-25cf', '2023-09-05cf', 
    '2023-09-15cf', '2023-09-19cf', '2023-09-26cf', '2023-10-17cf', 
    '2023-10-20cf', '2023-10-25cf', '2023-10-26cf', '2023-10-27cf', 
    '2023-11-1cf'
]

salt_plants = [
    '2022-05-12sal', '2022-09-08sal', '2022-10-08sal', '2022-10-19sal', 
    '2022-10-20sal', '2022-17-08sal', '2022-18-08sal', '2022-31-08sal', 
    '2023-03-06sal', '2023-03-07sal', '2023-03-21sal', '2023-03-22sal', 
    '2023-04-11sal', '2023-04-12sal', '2023-04-13sal', '2023-04-28sal'
]

fs = 62.5

# Design bandpass filter: 0.05 Hz to 2.0 Hz
sos = signal.butter(5, [0.05, 2.0], btype='bandpass', fs=fs, output='sos')

def load_raw_data(filepath):
    # Load and handle optional header row dynamically
    df = pd.read_csv(filepath, header=None)
    try:
        float(df.iloc[0, 0])
    except ValueError:
        # Header found, drop it
        df = df.iloc[1:].reset_index(drop=True)
    return df[0].values.astype(float)

def find_file(treatment, cond, ch, date_key):
    if treatment == 'cut_fire':
        dir_name = f'{cond}_c{ch}'
        dir_path = f'cut_fire/{cond}/raw/{dir_name}'
    else:
        dir_name = f'{cond}_Ch{ch}'
        dir_path = f'salt/{cond}/raw/{dir_name}'
    
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Directory not found: {dir_path}")
        
    files = os.listdir(dir_path)
    for f in files:
        # Extract exact plant ID by removing prefixes and splitting by underscore
        name = f.lower()
        for p in ['antes', 'depois']:
            if name.startswith(p):
                name = name[len(p):]
        name = name.split('_')[0]
        if name == date_key.lower():
            return os.path.join(dir_path, f)
            
    raise FileNotFoundError(f"No file matching {date_key} in {dir_path}")

all_data = []

print("Starting phase extraction...")

# Process Cut Fire group
print("\nProcessing Cut Fire group...")
for idx, plant in enumerate(cf_plants):
    print(f"Plant {idx+1}/{len(cf_plants)}: {plant}")
    for cond in ['before', 'after']:
        # Load raw data for all 4 channels
        y = {}
        for ch in [1, 2, 3, 4]:
            fp = find_file('cut_fire', cond, ch, plant)
            y[ch] = load_raw_data(fp)
            
        # Compute target time vector
        if cond == 'before':
            t_new = np.arange(-7200, 0, 1.0)
        else:
            t_new = np.arange(0, 10800, 1.0)
            
        phases = {}
        for ch in [1, 2, 3, 4]:
            y_ch = y[ch]
            N_ch = len(y_ch)
            
            # Compute channel-specific time vector relative to the condition
            if cond == 'before':
                t_orig = (np.arange(N_ch) - N_ch) / fs
            else:
                t_orig = np.arange(N_ch) / fs
                
            # Filter
            y_filt = signal.sosfiltfilt(sos, y_ch)
            # Hilbert
            z = signal.hilbert(y_filt)
            phi = np.angle(z)
            phi_unwrapped = np.unwrap(phi)
            # Interpolate to 1 Hz grid
            phases[ch] = np.interp(t_new, t_orig, phi_unwrapped)
            
        # Store in list
        df_cond = pd.DataFrame({
            'treatment': 'cut_fire',
            'plant_id': plant,
            'condition': cond,
            'time_sec': t_new,
            'C1_phase': phases[1],
            'C2_phase': phases[2],
            'C3_phase': phases[3],
            'C4_phase': phases[4]
        })
        all_data.append(df_cond)

# Process Salt group
print("\nProcessing Salt group...")
for idx, plant in enumerate(salt_plants):
    print(f"Plant {idx+1}/{len(salt_plants)}: {plant}")
    for cond in ['before', 'after']:
        # Load raw data for all 4 channels
        y = {}
        for ch in [1, 2, 3, 4]:
            fp = find_file('salt', cond, ch, plant)
            y[ch] = load_raw_data(fp)
            
        # Compute target time vector
        if cond == 'before':
            t_new = np.arange(-7200, 0, 1.0)
        else:
            t_new = np.arange(0, 10800, 1.0)
            
        phases = {}
        for ch in [1, 2, 3, 4]:
            y_ch = y[ch]
            N_ch = len(y_ch)
            
            # Compute channel-specific time vector relative to the condition
            if cond == 'before':
                t_orig = (np.arange(N_ch) - N_ch) / fs
            else:
                t_orig = np.arange(N_ch) / fs
                
            # Filter
            y_filt = signal.sosfiltfilt(sos, y_ch)
            # Hilbert
            z = signal.hilbert(y_filt)
            phi = np.angle(z)
            phi_unwrapped = np.unwrap(phi)
            # Interpolate to 1 Hz grid
            phases[ch] = np.interp(t_new, t_orig, phi_unwrapped)
            
        # Store in list
        df_cond = pd.DataFrame({
            'treatment': 'salt',
            'plant_id': plant,
            'condition': cond,
            'time_sec': t_new,
            'C1_phase': phases[1],
            'C2_phase': phases[2],
            'C3_phase': phases[3],
            'C4_phase': phases[4]
        })
        all_data.append(df_cond)

print("\nCombining all extracted phases and saving...")
final_df = pd.concat(all_data, ignore_index=True)
os.makedirs('csv', exist_ok=True)
final_df.to_csv('csv/extracted_phases.csv', index=False)
print("Phase extraction complete! Output saved to csv/extracted_phases.csv")
