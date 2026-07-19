import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import wilcoxon

def clean_source_file_cf(name):
    return name.split('_canal')[0].split('_c')[0]

def clean_source_file_salt(name):
    return name.split('_Ch')[0].split('_ch')[0].lower()

def conditional_mutual_information(X, Y, Z):
    # CMI: I(X; Y | Z)
    # Using the symmetric k-NN estimator to mitigate bias
    I_X_YZ = mutual_info_regression(np.column_stack([Y, Z]), X)[0]
    I_X_Z = mutual_info_regression(Z.reshape(-1, 1), X)[0]
    mi1 = max(0.0, I_X_YZ - I_X_Z)
    
    I_Y_XZ = mutual_info_regression(np.column_stack([X, Z]), Y)[0]
    I_Y_Z = mutual_info_regression(Z.reshape(-1, 1), Y)[0]
    mi2 = max(0.0, I_Y_XZ - I_Y_Z)
    
    return 0.5 * (mi1 + mi2)

def calculate_transfer_entropy(source, target, lag=1):
    # T_{S -> T} = I(T_future; S_past | T_past)
    if len(source) <= lag:
        return 0.0
        
    T_future = target[lag:]
    T_past = target[:-lag]
    S_past = source[:-lag]
    
    return conditional_mutual_information(T_future, S_past, T_past)

def main():
    print("Starting Transfer Entropy (TE) Analysis (Lags 1, 2, 3)...")
    np.random.seed(42)
    
    features = ['ApEn', 'DFA', 'PSD_mean']
    records = []
    
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
            
        plants = ch_data['C1']['plant_id'].unique()
        
        for plant in plants:
            plant_data = {}
            for ch in channels:
                plant_data[ch] = ch_data[ch][ch_data[ch]['plant_id'] == plant].reset_index(drop=True)
                
            for feat in features:
                for lag in [1, 2, 3]:
                    for follower in non_maestro_chs:
                        s_m = plant_data[maestro_ch][feat].values
                        s_f = plant_data[follower][feat].values
                        
                        # Direction 1: Maestro -> Follower
                        te_m_to_f = calculate_transfer_entropy(s_m, s_f, lag=lag)
                        
                        # Direction 2: Follower -> Maestro
                        te_f_to_m = calculate_transfer_entropy(s_f, s_m, lag=lag)
                        
                        records.append({
                            'treatment': treatment,
                            'plant_id': plant,
                            'feature': feat,
                            'lag': lag,
                            'follower_channel': follower,
                            'te_maestro_to_follower': te_m_to_f,
                            'te_follower_to_maestro': te_f_to_m
                        })

    df_te = pd.DataFrame(records)
    os.makedirs('csv', exist_ok=True)
    df_te.to_csv('csv/transfer_entropy_results.csv', index=False)
    print("Transfer Entropy results saved to csv/transfer_entropy_results.csv")
    
    # Statistical test of asymmetry: is TE(M->F) > TE(F->M)?
    print("\n=== Transfer Entropy Directional Asymmetry Test ===")
    for treatment in ['cut_fire', 'salt']:
        df_treat = df_te[df_te['treatment'] == treatment]
        for lag in [1, 2, 3]:
            df_sub = df_treat[df_treat['lag'] == lag]
            m_to_f = df_sub['te_maestro_to_follower'].values
            f_to_m = df_sub['te_follower_to_maestro'].values
            
            # Paired Wilcoxon test on asymmetry
            stat, p_val = wilcoxon(m_to_f, f_to_m, alternative='greater')
            print(f"{treatment.upper()} - Lag {lag}: Mean TE(M->F) = {np.mean(m_to_f):.4f}, Mean TE(F->M) = {np.mean(f_to_m):.4f}, p-value = {p_val:.5f}")
            
    # PLOT TRANSFER ENTROPY RESULTS
    plot_data = []
    for _, row in df_te.iterrows():
        plot_data.append({
            'Treatment': row['treatment'].replace('_', ' ').title(),
            'Feature': row['feature'],
            'Lag': f"Lag {row['lag']}",
            'Direction': 'Maestro → Followers',
            'TE (bits)': row['te_maestro_to_follower']
        })
        plot_data.append({
            'Treatment': row['treatment'].replace('_', ' ').title(),
            'Feature': row['feature'],
            'Lag': f"Lag {row['lag']}",
            'Direction': 'Followers → Maestro',
            'TE (bits)': row['te_follower_to_maestro']
        })
        
    df_plot = pd.DataFrame(plot_data)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    sns.set_theme(style="whitegrid")
    
    for idx, lag in enumerate([1, 2, 3]):
        ax = axes[idx]
        df_lag = df_plot[df_plot['Lag'] == f"Lag {lag}"]
        
        sns.barplot(
            data=df_lag,
            x='Feature',
            y='TE (bits)',
            hue='Direction',
            ax=ax,
            palette={'Maestro → Followers': '#d62728', 'Followers → Maestro': '#7f7f7f'},
            edgecolor='black',
            alpha=0.85,
            errorbar='se'
        )
        
        ax.set_title(f"Transfer Entropy (Lag {lag})", fontsize=14, fontweight='bold')
        ax.set_xlabel("Complexity Feature", fontsize=12)
        if idx == 0:
            ax.set_ylabel("Information Flow (bits)", fontsize=12)
        else:
            ax.set_ylabel("")
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # Legend configuration
        if idx == 2:
            ax.legend(title='Information Flow', title_fontsize='11', fontsize='10', loc='upper right')
        else:
            ax.get_legend().remove()
            
    plt.suptitle("Information Flow Directionality: Transfer Entropy comparison (Lags 1, 2, 3)", fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    plot_path = "img/transfer_entropy_directionality.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Transfer entropy plot saved to {plot_path}")

if __name__ == '__main__':
    main()
