import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression

def clean_source_file_cf(name):
    return name.split('_canal')[0].split('_c')[0]

def clean_source_file_salt(name):
    return name.split('_Ch')[0].split('_ch')[0].lower()

def conditional_mutual_information(X, Y, Z):
    # Estimate I(X; Y | Z) = I(X; (Y, Z)) - I(X; Z)
    # Average with symmetric estimator for stability
    # Estimate 1: I(X; Y | Z)
    I_X_YZ = mutual_info_regression(np.column_stack([Y, Z]), X)[0]
    I_X_Z = mutual_info_regression(Z.reshape(-1, 1), X)[0]
    mi1 = max(0.0, I_X_YZ - I_X_Z)
    
    # Estimate 2: I(Y; X | Z)
    I_Y_XZ = mutual_info_regression(np.column_stack([X, Z]), Y)[0]
    I_Y_Z = mutual_info_regression(Z.reshape(-1, 1), Y)[0]
    mi2 = max(0.0, I_Y_XZ - I_Y_Z)
    
    return 0.5 * (mi1 + mi2)

def pairwise_mutual_information(X, Y):
    # Estimate I(X; Y) and average for symmetry
    mi1 = mutual_info_regression(X.reshape(-1, 1), Y)[0]
    mi2 = mutual_info_regression(Y.reshape(-1, 1), X)[0]
    return 0.5 * (mi1 + mi2)

def main():
    # Set random seed for reproducibility in surrogate shuffling
    np.random.seed(42)
    
    features_to_analyze = ['ApEn', 'DFA', 'PSD_mean']
    records = []
    
    for treatment in ['cut_fire', 'salt']:
        print(f"\nProcessing Conditional Information and Surrogate Test for {treatment}...")
        maestro_ch = 'C3' if treatment == 'cut_fire' else 'C1'
        channels = ['C1', 'C2', 'C3', 'C4']
        non_maestro_chs = [ch for ch in channels if ch != maestro_ch]
        
        for cond in ['before', 'after']:
            print(f"  Condition: {cond}...")
            # Load and align all channels for this treatment and condition
            ch_data = {}
            for ch in channels:
                if treatment == 'cut_fire':
                    file_path = f'cut_fire/{cond}/features/df_{cond}_c{ch[1]}.csv'
                else:
                    file_path = f'salt/{cond}/features/df_{cond}_Ch{ch[1]}.csv'
                    
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Feature file not found: {file_path}")
                    
                df_ch = pd.read_csv(file_path)
                
                # Clean source_file to get aligned plant_id
                if treatment == 'cut_fire':
                    df_ch['plant_id'] = df_ch['source_file'].apply(clean_source_file_cf)
                else:
                    df_ch['plant_id'] = df_ch['source_file'].apply(clean_source_file_salt)
                    
                # Sort and reset index to ensure exact alignment
                df_ch = df_ch.sort_values(['plant_id', 'minute']).reset_index(drop=True)
                ch_data[ch] = df_ch
                
            # Verify alignment
            for ch in channels[1:]:
                if not (ch_data[channels[0]]['plant_id'] == ch_data[ch]['plant_id']).all():
                    raise ValueError(f"Alignment mismatch in plant_id for {treatment} {cond} channel {ch}")
                if not (ch_data[channels[0]]['minute'] == ch_data[ch]['minute']).all():
                    raise ValueError(f"Alignment mismatch in minute for {treatment} {cond} channel {ch}")
            
            # Perform Information Theory computations for each feature
            for feat in features_to_analyze:
                # Extract series
                X = {ch: ch_data[ch][feat].values for ch in channels}
                
                # Create surrogate: Shuffled maestro channel (breaks temporal coupling)
                Z_shuffled = np.random.permutation(X[maestro_ch])
                
                # 1. Unconditioned Pairwise MI
                mi_unc_matrix = pd.DataFrame(np.zeros((4, 4)), index=channels, columns=channels)
                for i in range(4):
                    for j in range(i, 4):
                        ch_a = channels[i]
                        ch_b = channels[j]
                        if i == j:
                            mi_val = 0.0
                        else:
                            mi_val = pairwise_mutual_information(X[ch_a], X[ch_b])
                        mi_unc_matrix.loc[ch_a, ch_b] = mi_val
                        mi_unc_matrix.loc[ch_b, ch_a] = mi_val
                        
                # 2. Conditioned MI (conditioned on true maestro)
                mi_cond_matrix = pd.DataFrame(np.zeros((3, 3)), index=non_maestro_chs, columns=non_maestro_chs)
                # 3. Surrogate Conditioned MI (conditioned on shuffled maestro)
                mi_surr_matrix = pd.DataFrame(np.zeros((3, 3)), index=non_maestro_chs, columns=non_maestro_chs)
                
                for i in range(3):
                    for j in range(i, 3):
                        ch_a = non_maestro_chs[i]
                        ch_b = non_maestro_chs[j]
                        if i == j:
                            mi_val = 0.0
                            mi_surr_val = 0.0
                        else:
                            mi_val = conditional_mutual_information(X[ch_a], X[ch_b], X[maestro_ch])
                            mi_surr_val = conditional_mutual_information(X[ch_a], X[ch_b], Z_shuffled)
                        mi_cond_matrix.loc[ch_a, ch_b] = mi_val
                        mi_cond_matrix.loc[ch_b, ch_a] = mi_val
                        mi_surr_matrix.loc[ch_a, ch_b] = mi_surr_val
                        mi_surr_matrix.loc[ch_b, ch_a] = mi_surr_val
                        
                # Store records for CSV
                for ch_a in channels:
                    for ch_b in channels:
                        if ch_a < ch_b:
                            mi_cond_val = None
                            mi_surr_val = None
                            if ch_a in non_maestro_chs and ch_b in non_maestro_chs:
                                mi_cond_val = mi_cond_matrix.loc[ch_a, ch_b]
                                mi_surr_val = mi_surr_matrix.loc[ch_a, ch_b]
                                
                            records.append({
                                'treatment': treatment,
                                'condition': cond,
                                'feature': feat,
                                'pair': f"{ch_a}-{ch_b}",
                                'mi_unconditioned': mi_unc_matrix.loc[ch_a, ch_b],
                                'mi_conditioned_by_maestro': mi_cond_val,
                                'mi_conditioned_by_shuffled_maestro': mi_surr_val
                            })
                            
            # Keep matrices for combined plotting
            if cond == 'before':
                ch_data_before = ch_data
            
        # Combined plotting for before vs after
        for feat in features_to_analyze:
            X_bef = {ch: ch_data_before[ch][feat].values for ch in channels}
            mi_unc_bef = pd.DataFrame(np.zeros((4, 4)), index=channels, columns=channels)
            mi_cond_bef = pd.DataFrame(np.zeros((3, 3)), index=non_maestro_chs, columns=non_maestro_chs)
            for i in range(4):
                for j in range(i, 4):
                    ch_a, ch_b = channels[i], channels[j]
                    if i != j:
                        val = pairwise_mutual_information(X_bef[ch_a], X_bef[ch_b])
                        mi_unc_bef.loc[ch_a, ch_b] = val
                        mi_unc_bef.loc[ch_b, ch_a] = val
            for i in range(3):
                for j in range(i, 3):
                    ch_a, ch_b = non_maestro_chs[i], non_maestro_chs[j]
                    if i != j:
                        val = conditional_mutual_information(X_bef[ch_a], X_bef[ch_b], X_bef[maestro_ch])
                        mi_cond_bef.loc[ch_a, ch_b] = val
                        mi_cond_bef.loc[ch_b, ch_a] = val
                        
            X_aft = {ch: ch_data[ch][feat].values for ch in channels}
            mi_unc_aft = pd.DataFrame(np.zeros((4, 4)), index=channels, columns=channels)
            mi_cond_aft = pd.DataFrame(np.zeros((3, 3)), index=non_maestro_chs, columns=non_maestro_chs)
            for i in range(4):
                for j in range(i, 4):
                    ch_a, ch_b = channels[i], channels[j]
                    if i != j:
                        val = pairwise_mutual_information(X_aft[ch_a], X_aft[ch_b])
                        mi_unc_aft.loc[ch_a, ch_b] = val
                        mi_unc_aft.loc[ch_b, ch_a] = val
            for i in range(3):
                for j in range(i, 3):
                    ch_a, ch_b = non_maestro_chs[i], non_maestro_chs[j]
                    if i != j:
                        val = conditional_mutual_information(X_aft[ch_a], X_aft[ch_b], X_aft[maestro_ch])
                        mi_cond_aft.loc[ch_a, ch_b] = val
                        mi_cond_aft.loc[ch_b, ch_a] = val
            
            # Setup 2x2 subplot
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            
            vmax_unc = max(mi_unc_bef.max().max(), mi_unc_aft.max().max(), 0.05)
            vmax_cond = max(mi_cond_bef.max().max(), mi_cond_aft.max().max(), 0.05)
            
            mask_unc = np.eye(4, dtype=bool)
            mask_cond = np.eye(3, dtype=bool)
            
            # Row 1: Unconditioned MI
            sns.heatmap(mi_unc_bef, annot=True, fmt=".3f", cmap="viridis", mask=mask_unc, vmin=0, vmax=vmax_unc,
                        ax=axes[0, 0], cbar_kws={'label': 'MI (bits)'})
            axes[0, 0].set_title(f"Before Stress: Pairwise MI I(Ci; Cj)", fontsize=13, fontweight='semibold')
            
            sns.heatmap(mi_unc_aft, annot=True, fmt=".3f", cmap="viridis", mask=mask_unc, vmin=0, vmax=vmax_unc,
                        ax=axes[0, 1], cbar_kws={'label': 'MI (bits)'})
            axes[0, 1].set_title(f"After Stress: Pairwise MI I(Ci; Cj)", fontsize=13, fontweight='semibold')
            
            # Row 2: Conditioned MI
            sns.heatmap(mi_cond_bef, annot=True, fmt=".3f", cmap="plasma", mask=mask_cond, vmin=0, vmax=vmax_cond,
                        ax=axes[1, 0], cbar_kws={'label': 'Cond. MI (bits)'})
            axes[1, 0].set_title(f"Before Stress: Cond. MI I(Ci; Cj | {maestro_ch})", fontsize=13, fontweight='semibold')
            
            sns.heatmap(mi_cond_aft, annot=True, fmt=".3f", cmap="plasma", mask=mask_cond, vmin=0, vmax=vmax_cond,
                        ax=axes[1, 1], cbar_kws={'label': 'Cond. MI (bits)'})
            axes[1, 1].set_title(f"After Stress: Cond. MI I(Ci; Cj | {maestro_ch})", fontsize=13, fontweight='semibold')
            
            # Main Title
            plt.suptitle(f"Mutual Information Heatmaps ({feat}): {treatment.replace('_', ' ').title()}", 
                         fontsize=16, fontweight='bold', y=0.98)
            plt.tight_layout()
            
            # Save
            plot_path = f"img/{treatment}_mi_heatmaps_{feat}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved heatmap plot for {feat} to {plot_path}")
            
    # Save the records to CSV
    mi_df = pd.DataFrame(records)
    mi_df.to_csv('csv/conditional_mi_results.csv', index=False)
    print("\nConditional Mutual Information results saved to csv/conditional_mi_results.csv")
    
    # Print a summary of changes including surrogate results
    print("\n=== Mean Mutual Information & Surrogate Summary ===")
    summary = mi_df.groupby(['treatment', 'condition', 'feature'])[[
        'mi_unconditioned', 
        'mi_conditioned_by_maestro', 
        'mi_conditioned_by_shuffled_maestro'
    ]].mean()
    print(summary)

if __name__ == '__main__':
    main()
