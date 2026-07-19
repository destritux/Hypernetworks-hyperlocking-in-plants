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
    I_X_YZ = mutual_info_regression(np.column_stack([Y, Z]), X)[0]
    I_X_Z = mutual_info_regression(Z.reshape(-1, 1), X)[0]
    mi1 = max(0.0, I_X_YZ - I_X_Z)
    
    I_Y_XZ = mutual_info_regression(np.column_stack([X, Z]), Y)[0]
    I_Y_Z = mutual_info_regression(Z.reshape(-1, 1), Y)[0]
    mi2 = max(0.0, I_Y_XZ - I_Y_Z)
    
    return 0.5 * (mi1 + mi2)

def pairwise_mutual_information(X, Y):
    mi1 = mutual_info_regression(X.reshape(-1, 1), Y)[0]
    mi2 = mutual_info_regression(Y.reshape(-1, 1), X)[0]
    return 0.5 * (mi1 + mi2)

def main():
    print("Starting Advanced CMI Analysis: 15-min Sliding Windows and Label Permutation...")
    np.random.seed(42)
    
    features = ['ApEn', 'DFA', 'PSD_mean']
    window_duration = 15 # minutes (changed from 30)
    n_windows = 180 // window_duration # 12 windows
    
    sliding_window_records = []
    null_model_records = []
    
    for treatment in ['cut_fire', 'salt']:
        maestro_ch = 'C3' if treatment == 'cut_fire' else 'C1'
        channels = ['C1', 'C2', 'C3', 'C4']
        non_maestro_chs = [ch for ch in channels if ch != maestro_ch]
        
        # Load AFTER condition feature datasets
        ch_data = {}
        for ch in channels:
            if treatment == 'cut_fire':
                file_path = f'cut_fire/after/features/df_after_c{ch[1]}.csv'
            else:
                file_path = f'salt/after/features/df_after_Ch{ch[1]}.csv'
                
            df_ch = pd.read_csv(file_path)
            if treatment == 'cut_fire':
                df_ch['plant_id'] = df_ch['source_file'].apply(clean_source_file_cf)
            else:
                df_ch['plant_id'] = df_ch['source_file'].apply(clean_source_file_salt)
                
            df_ch = df_ch.sort_values(['plant_id', 'minute']).reset_index(drop=True)
            ch_data[ch] = df_ch
            
        # 1. SLIDING WINDOW CMI ANALYSIS
        print(f"  Computing 15-min sliding windows for {treatment}...")
        for w_idx in range(n_windows):
            start_min = w_idx * window_duration
            end_min = (w_idx + 1) * window_duration
            
            # Slice features for this window
            ch_window = {}
            for ch in channels:
                ch_window[ch] = ch_data[ch][(ch_data[ch]['minute'] >= start_min) & (ch_data[ch]['minute'] < end_min)]
                
            # Loop over features
            for feat in features:
                # Extract series
                X = {ch: ch_window[ch][feat].values for ch in channels}
                Z_shuffled = np.random.permutation(X[maestro_ch])
                
                # Compute CMI between followers, conditioned on real maestro and shuffled maestro
                mi_uncs = []
                cmi_reals = []
                cmi_shuffs = []
                
                for i in range(len(non_maestro_chs)):
                    for j in range(i+1, len(non_maestro_chs)):
                        ch_a = non_maestro_chs[i]
                        ch_b = non_maestro_chs[j]
                        
                        mi_uncs.append(pairwise_mutual_information(X[ch_a], X[ch_b]))
                        cmi_reals.append(conditional_mutual_information(X[ch_a], X[ch_b], X[maestro_ch]))
                        cmi_shuffs.append(conditional_mutual_information(X[ch_a], X[ch_b], Z_shuffled))
                        
                sliding_window_records.append({
                    'treatment': treatment,
                    'window_start': start_min,
                    'window_end': end_min,
                    'feature': feat,
                    'mi_unconditioned': np.mean(mi_uncs),
                    'mi_conditioned_by_maestro': np.mean(cmi_reals),
                    'mi_conditioned_by_shuffled_maestro': np.mean(cmi_shuffs)
                })
                
        # 2. LABEL PERMUTATION NULL MODEL (Conditioning on FALSE maestros)
        print(f"  Computing label permutation null model for {treatment}...")
        for feat in features:
            X_full = {ch: ch_data[ch][feat].values for ch in channels}
            
            # Condition on the true maestro
            cmi_true = []
            for i in range(len(non_maestro_chs)):
                for j in range(i+1, len(non_maestro_chs)):
                    ch_a, ch_b = non_maestro_chs[i], non_maestro_chs[j]
                    cmi_true.append(conditional_mutual_information(X_full[ch_a], X_full[ch_b], X_full[maestro_ch]))
            
            # Condition on each false maestro
            for false_maestro in non_maestro_chs:
                other_chs = [ch for ch in channels if ch != false_maestro]
                cmi_false = []
                
                for i in range(len(other_chs)):
                    for j in range(i+1, len(other_chs)):
                        ch_a, ch_b = other_chs[i], other_chs[j]
                        cmi_false.append(conditional_mutual_information(X_full[ch_a], X_full[ch_b], X_full[false_maestro]))
                        
                null_model_records.append({
                    'treatment': treatment,
                    'feature': feat,
                    'maestro_type': 'True Maestro' if false_maestro == maestro_ch else 'False Maestro',
                    'maestro_channel': false_maestro,
                    'cmi_value': np.mean(cmi_false)
                })
                
            null_model_records.append({
                'treatment': treatment,
                'feature': feat,
                'maestro_type': 'True Maestro',
                'maestro_channel': maestro_ch,
                'cmi_value': np.mean(cmi_true)
            })

    # Save to CSVs
    df_window = pd.DataFrame(sliding_window_records)
    df_null = pd.DataFrame(null_model_records)
    
    os.makedirs('csv', exist_ok=True)
    df_window.to_csv('csv/sliding_window_cmi.csv', index=False)
    df_null.to_csv('csv/null_model_label_permutation.csv', index=False)
    print("Advanced CMI data saved in csv/sliding_window_cmi.csv and csv/null_model_label_permutation.csv")
    
    # 3. PLOTTING ADVANCED ANALYSES (2x2 Panel Plot)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    feat_rep = 'ApEn'
    
    # Subplot 0,0: Cut/Fire sliding window
    df_w_cf = df_window[(df_window['treatment'] == 'cut_fire') & (df_window['feature'] == feat_rep)]
    time_bins = [f"{r['window_start']}-{r['window_end']}m" for _, r in df_w_cf.iterrows()]
    
    axes[0, 0].plot(time_bins, df_w_cf['mi_unconditioned'], marker='o', label="MI unconditioned (I(Ci;Cj))", color='#2ca02c', lw=2.5)
    axes[0, 0].plot(time_bins, df_w_cf['mi_conditioned_by_maestro'], marker='s', label="CMI condicionada no maestro real (I(Ci;Cj | Maestro))", color='#d62728', lw=2.5)
    axes[0, 0].plot(time_bins, df_w_cf['mi_conditioned_by_shuffled_maestro'], marker='^', label="CMI condicionada no maestro shufflado (I(Ci;Cj | Maestro_shuff))", color='#1f77b4', lw=2.5, ls='--')
    axes[0, 0].set_title(f"Cut Fire: 15-min Sliding Window CMI ({feat_rep})", fontsize=13, fontweight='bold')
    axes[0, 0].set_xlabel("Time Post-Stress (min)", fontsize=11)
    axes[0, 0].set_ylabel("Shared Information (bits)", fontsize=11)
    axes[0, 0].set_xticklabels(time_bins, rotation=45, ha='right')
    axes[0, 0].grid(True, linestyle=':', alpha=0.6)
    axes[0, 0].legend(fontsize=9)
    
    # Subplot 0,1: Salt sliding window
    df_w_salt = df_window[(df_window['treatment'] == 'salt') & (df_window['feature'] == feat_rep)]
    axes[0, 1].plot(time_bins, df_w_salt['mi_unconditioned'], marker='o', label="MI unconditioned (I(Ci;Cj))", color='#2ca02c', lw=2.5)
    axes[0, 1].plot(time_bins, df_w_salt['mi_conditioned_by_maestro'], marker='s', label="CMI condicionada no maestro real (I(Ci;Cj | Maestro))", color='#d62728', lw=2.5)
    axes[0, 1].plot(time_bins, df_w_salt['mi_conditioned_by_shuffled_maestro'], marker='^', label="CMI condicionada no maestro shufflado (I(Ci;Cj | Maestro_shuff))", color='#1f77b4', lw=2.5, ls='--')
    axes[0, 1].set_title(f"Salt: 15-min Sliding Window CMI ({feat_rep})", fontsize=13, fontweight='bold')
    axes[0, 1].set_xlabel("Time Post-Stress (min)", fontsize=11)
    axes[0, 1].set_ylabel("Shared Information (bits)", fontsize=11)
    axes[0, 1].set_xticklabels(time_bins, rotation=45, ha='right')
    axes[0, 1].grid(True, linestyle=':', alpha=0.6)
    axes[0, 1].legend(fontsize=9)
    
    # Subplots 1,0 and 1,1: Label Permutation Null Model for both treatments (average across all features)
    df_n_cf = df_null[df_null['treatment'] == 'cut_fire'].groupby(['maestro_channel', 'maestro_type'])['cmi_value'].mean().reset_index()
    df_n_cf = df_n_cf.sort_values('maestro_type', ascending=False)
    
    colors_cf = ['#d62728' if t == 'True Maestro' else '#7f7f7f' for t in df_n_cf['maestro_type']]
    bars_cf = axes[1, 0].bar(df_n_cf['maestro_channel'], df_n_cf['cmi_value'], color=colors_cf, width=0.4, edgecolor='black', alpha=0.85)
    axes[1, 0].set_title("Cut Fire: Label Permutation Null Model", fontsize=13, fontweight='bold')
    axes[1, 0].set_xlabel("Conditioning Channel (Potential Maestro)", fontsize=11)
    axes[1, 0].set_ylabel("Average CMI (bits)", fontsize=11)
    axes[1, 0].grid(True, linestyle=':', alpha=0.6, axis='y')
    for bar in bars_cf:
        axes[1, 0].text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 0.005, f"{bar.get_height():.3f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
        
    df_n_salt = df_null[df_null['treatment'] == 'salt'].groupby(['maestro_channel', 'maestro_type'])['cmi_value'].mean().reset_index()
    df_n_salt = df_n_salt.sort_values('maestro_type', ascending=False)
    
    colors_salt = ['#d62728' if t == 'True Maestro' else '#7f7f7f' for t in df_n_salt['maestro_type']]
    bars_salt = axes[1, 1].bar(df_n_salt['maestro_channel'], df_n_salt['cmi_value'], color=colors_salt, width=0.4, edgecolor='black', alpha=0.85)
    axes[1, 1].set_title("Salt: Label Permutation Null Model", fontsize=13, fontweight='bold')
    axes[1, 1].set_xlabel("Conditioning Channel (Potential Maestro)", fontsize=11)
    axes[1, 1].set_ylabel("Average CMI (bits)", fontsize=11)
    axes[1, 1].grid(True, linestyle=':', alpha=0.6, axis='y')
    for bar in bars_salt:
        axes[1, 1].text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 0.005, f"{bar.get_height():.3f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
        
    plt.suptitle("Advanced CMI: 15-min Temporal Sensitivity and Label Permutation Null Model", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    plot_path = "img/advanced_cmi_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Advanced CMI analysis plot saved to {plot_path}")

if __name__ == '__main__':
    main()
