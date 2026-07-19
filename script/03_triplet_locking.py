import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    print("Loading extracted phases...")
    if not os.path.exists('csv/extracted_phases.csv'):
        raise FileNotFoundError("csv/extracted_phases.csv not found! Run 01_extract_phase.py first.")
        
    df = pd.read_csv('csv/extracted_phases.csv')
    
    # We will test two main hypotheses for the maestro node:
    # Hypothesis 1: Stem C1 modula as folhas C2 e C4 -> theta = phi2 + phi4 - 2*phi1
    # Hypothesis 2: Folha C3 modula o caule C1 e a folha C4 -> theta = phi1 + phi4 - 2*phi3
    
    hypotheses = {
        'stem_modulates': {
            'formula': r'$\theta_{tri} = \phi_2 + \phi_4 - 2\phi_1$',
            'func': lambda row: row['C2_phase'] + row['C4_phase'] - 2 * row['C1_phase']
        },
        'stressed_leaf_modulates': {
            'formula': r'$\theta_{tri} = \phi_1 + \phi_4 - 2\phi_3$',
            'func': lambda row: row['C1_phase'] + row['C4_phase'] - 2 * row['C3_phase']
        }
    }
    
    tli_records = []
    
    for treatment in ['cut_fire', 'salt']:
        print(f"\nProcessing triplet locking for {treatment}...")
        df_treat = df[df['treatment'] == treatment]
        plant_ids = df_treat['plant_id'].unique()
        
        # Time grid for plotting
        time_sec = np.concatenate([np.arange(-7200, 0, 1.0), np.arange(0, 10800, 1.0)])
        time_min = time_sec / 60.0
        
        # Collect continuous triplet phase series for each hypothesis
        hyp_series = {hyp_name: [] for hyp_name in hypotheses}
        
        for plant in plant_ids:
            df_plant = df_treat[df_treat['plant_id'] == plant].sort_values('time_sec')
            
            df_before = df_plant[df_plant['condition'] == 'before'].sort_values('time_sec')
            df_after = df_plant[df_plant['condition'] == 'after'].sort_values('time_sec')
            
            if len(df_before) != 7200 or len(df_after) != 10800:
                continue
                
            for hyp_name, hyp_info in hypotheses.items():
                # Compute triplet phase for before and after
                theta_bef = hyp_info['func'](df_before).values
                theta_aft = hyp_info['func'](df_after).values
                
                # Compute Triplet Locking Index (TLI)
                tli_bef = np.abs(np.mean(np.exp(1j * theta_bef)))
                tli_aft = np.abs(np.mean(np.exp(1j * theta_aft)))
                
                tli_records.append({
                    'treatment': treatment,
                    'plant_id': plant,
                    'hypothesis': hyp_name,
                    'tli_before': tli_bef,
                    'tli_after': tli_aft
                })
                
                # Compute continuous normalized triplet phase dynamically
                theta_full = np.concatenate([theta_bef, theta_aft])
                idx_zero = np.argmin(np.abs(time_sec))
                theta_norm = theta_full - theta_full[idx_zero]
                hyp_series[hyp_name].append(theta_norm)
                
        # Generate side-by-side plot comparing hypotheses for this treatment
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
        color_before = '#1f77b4' # steel blue
        color_after = '#d62728'  # crimson red
        
        for idx, (hyp_name, hyp_info) in enumerate(hypotheses.items()):
            ax = axes[idx]
            series_array = np.array(hyp_series[hyp_name])
            
            if len(series_array) == 0:
                continue
                
            mean_series = np.mean(series_array, axis=0)
            std_series = np.std(series_array, axis=0)
            
            # Split time and values into before and after dynamically
            idx_zero = np.argmin(np.abs(time_sec))
            t_bef = time_min[:idx_zero]
            mean_bef = mean_series[:idx_zero]
            std_bef = std_series[:idx_zero]
            
            t_aft = time_min[idx_zero:]
            mean_aft = mean_series[idx_zero:]
            std_aft = std_series[idx_zero:]
            
            # Plot before and after
            ax.plot(t_bef, mean_bef, color=color_before, lw=2, label='Baseline (Before)')
            ax.fill_between(t_bef, mean_bef - std_bef, mean_bef + std_bef, color=color_before, alpha=0.2)
            
            ax.plot(t_aft, mean_aft, color=color_after, lw=2, label='Post-Stress (After)')
            ax.fill_between(t_aft, mean_aft - std_aft, mean_aft + std_aft, color=color_after, alpha=0.2)
            
            ax.axvline(x=0, color='black', linestyle='--', lw=1.5, label='Stress Moment')
            
            # Title and labels
            title_name = "Stem (C1) as Modulator" if hyp_name == 'stem_modulates' else "Stressed Leaf (C3) as Modulator"
            ax.set_title(f"{title_name}\n{hyp_info['formula']}", fontsize=14, fontweight='semibold')
            ax.set_xlabel("Time relative to stress (minutes)", fontsize=12)
            ax.set_ylabel(r"Normalized $\theta_{tri}$ (rad)", fontsize=12)
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend(fontsize=10, loc='upper left')
            
        plt.suptitle(f"Triplet Phase Locking (Hyperlocking) Evolution: {treatment.replace('_', ' ').title()}", fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        os.makedirs('img', exist_ok=True)
        plot_path = f"img/{treatment}_triplet_drift.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved triplet drift plot to {plot_path}")
        
    # Save TLI records to CSV
    tli_df = pd.DataFrame(tli_records)
    tli_df.to_csv('csv/triplet_locking.csv', index=False)
    print("\nTriplet Locking Indices saved to csv/triplet_locking.csv")
    
    # Print summary of TLI before and after to verify hyperlocking
    print("\n=== Mean TLI Summary (Before vs After) ===")
    summary = tli_df.groupby(['treatment', 'hypothesis'])[['tli_before', 'tli_after']].mean()
    print(summary)

if __name__ == '__main__':
    main()
