import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    print("Loading extracted phases...")
    if not os.path.exists('csv/extracted_phases.csv'):
        raise FileNotFoundError("csv/extracted_phases.csv not found! Run 01_extract_phase.py first.")
        
    df = pd.read_csv('csv/extracted_phases.csv')
    
    pairs = [
        ('C1', 'C2'),
        ('C1', 'C3'),
        ('C1', 'C4'),
        ('C2', 'C3'),
        ('C2', 'C4'),
        ('C3', 'C4')
    ]
    
    plv_records = []
    
    # We will process each treatment group separately
    for treatment in ['cut_fire', 'salt']:
        print(f"\nProcessing pairwise synchronization for {treatment}...")
        df_treat = df[df['treatment'] == treatment]
        plant_ids = df_treat['plant_id'].unique()
        
        # We will collect the normalized phase differences for all plants to plot the mean and dispersion
        # Time grid: before has 7200 samples, after has 10800 samples. Total 18000 samples.
        time_sec = np.concatenate([np.arange(-7200, 0, 1.0), np.arange(0, 10800, 1.0)])
        time_min = time_sec / 60.0
        
        # A dictionary to hold the normalized phase difference series for each pair:
        # key: pair, value: array of shape (num_plants, 18000)
        pair_diffs = {pair: [] for pair in pairs}
        
        for plant in plant_ids:
            df_plant = df_treat[df_treat['plant_id'] == plant].sort_values('time_sec')
            
            # Separate before and after
            df_before = df_plant[df_plant['condition'] == 'before'].sort_values('time_sec')
            df_after = df_plant[df_plant['condition'] == 'after'].sort_values('time_sec')
            
            # Check length to ensure completeness
            if len(df_before) != 7200 or len(df_after) != 10800:
                print(f"Warning: plant {plant} has incomplete samples (before={len(df_before)}, after={len(df_after)}). Skipping PLV for this plant.")
                continue
                
            for pair in pairs:
                ch_i, ch_j = pair
                phi_i_bef = df_before[f'{ch_i}_phase'].values
                phi_j_bef = df_before[f'{ch_j}_phase'].values
                
                phi_i_aft = df_after[f'{ch_i}_phase'].values
                phi_j_aft = df_after[f'{ch_j}_phase'].values
                
                # Calculate PLV before and after
                plv_bef = np.abs(np.mean(np.exp(1j * (phi_i_bef - phi_j_bef))))
                plv_aft = np.abs(np.mean(np.exp(1j * (phi_i_aft - phi_j_aft))))
                
                plv_records.append({
                    'treatment': treatment,
                    'plant_id': plant,
                    'pair': f"{ch_i}-{ch_j}",
                    'plv_before': plv_bef,
                    'plv_after': plv_aft
                })
                
                # Concatenate the full series to calculate continuous phase difference drift
                phi_i_full = np.concatenate([phi_i_bef, phi_i_aft])
                phi_j_full = np.concatenate([phi_j_bef, phi_j_aft])
                diff_full = phi_i_full - phi_j_full
                
                # Normalize at t = 0 dynamically
                idx_zero = np.argmin(np.abs(time_sec))
                diff_norm = diff_full - diff_full[idx_zero]
                pair_diffs[pair].append(diff_norm)
                
        # Generate the plot for this treatment
        fig, axes = plt.subplots(3, 2, figsize=(14, 18), sharex=True)
        axes = axes.flatten()
        
        # Color palette for professional layout
        color_before = '#1f77b4' # steel blue
        color_after = '#d62728'  # crimson red
        
        for idx, pair in enumerate(pairs):
            ax = axes[idx]
            pair_name = f"{pair[0]} - {pair[1]}"
            
            # Convert list of arrays to 2D numpy array: (num_plants, 18000)
            diffs_array = np.array(pair_diffs[pair])
            
            if len(diffs_array) == 0:
                continue
                
            # Calculate mean and std at each time point
            mean_diff = np.mean(diffs_array, axis=0)
            std_diff = np.std(diffs_array, axis=0)
            
            # Split time and values into before (0 to idx_zero-1) and after (idx_zero onwards)
            idx_zero = np.argmin(np.abs(time_sec))
            t_bef = time_min[:idx_zero]
            mean_bef = mean_diff[:idx_zero]
            std_bef = std_diff[:idx_zero]
            
            t_aft = time_min[idx_zero:]
            mean_aft = mean_diff[idx_zero:]
            std_aft = std_diff[idx_zero:]
            
            # Plot before phase difference
            ax.plot(t_bef, mean_bef, color=color_before, lw=2, label='Baseline (Before)')
            ax.fill_between(t_bef, mean_bef - std_bef, mean_bef + std_bef, color=color_before, alpha=0.2)
            
            # Plot after phase difference
            ax.plot(t_aft, mean_aft, color=color_after, lw=2, label='Post-Stress (After)')
            ax.fill_between(t_aft, mean_aft - std_aft, mean_aft + std_aft, color=color_after, alpha=0.2)
            
            # Vertical line indicating the stress onset
            ax.axvline(x=0, color='black', linestyle='--', lw=1.5, label='Stress Moment')
            
            ax.set_title(f"Pairwise Phase Difference: {pair_name}", fontsize=14, fontweight='semibold')
            ax.set_ylabel(r"$\Delta\phi$ (rad)", fontsize=12)
            ax.grid(True, linestyle=':', alpha=0.6)
            
            if idx in [4, 5]:
                ax.set_xlabel("Time relative to stress (minutes)", fontsize=12)
                
            if idx == 0:
                ax.legend(fontsize=10, loc='upper left')
                
        plt.tight_layout()
        os.makedirs('img', exist_ok=True)
        plot_path = f"img/{treatment}_pairwise_drift.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved drift plot to {plot_path}")
        
    # Save the PLV records to CSV
    plv_df = pd.DataFrame(plv_records)
    plv_df.to_csv('csv/plv_results.csv', index=False)
    print("\nPairwise Locking values saved to csv/plv_results.csv")
    
    # Print summary of PLVs before and after to verify the absence of pairwise locking
    print("\n=== Mean PLV Summary (Before vs After) ===")
    summary = plv_df.groupby(['treatment', 'pair'])[['plv_before', 'plv_after']].mean()
    print(summary)

if __name__ == '__main__':
    main()
