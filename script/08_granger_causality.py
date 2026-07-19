import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import grangercausalitytests

def clean_source_file_cf(name):
    return name.split('_canal')[0].split('_c')[0]

def clean_source_file_salt(name):
    return name.split('_Ch')[0].split('_ch')[0].lower()

def run_granger_test(y, x, lag=1):
    data = np.column_stack([y, x])
    try:
        res = grangercausalitytests(data, maxlag=[lag], verbose=False)
        p_val = res[lag][0]['ssr_ftest'][1]
        return p_val
    except Exception as e:
        return 1.0

def main():
    print("Starting Granger Causality Directionality Analysis (Lags 1, 2, 3)...")
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
            
        # Get unique plant IDs
        plants = ch_data['C1']['plant_id'].unique()
        
        for plant in plants:
            plant_data = {}
            for ch in channels:
                plant_data[ch] = ch_data[ch][ch_data[ch]['plant_id'] == plant].reset_index(drop=True)
                
            for feat in features:
                # Difference the series to ensure stationarity
                diff_series = {}
                for ch in channels:
                    y_ch = plant_data[ch][feat].values
                    diff_series[ch] = np.diff(y_ch)
                    
                # Test causality for lags 1, 2, and 3
                for lag in [1, 2, 3]:
                    for follower in non_maestro_chs:
                        # Direction 1: Maestro -> Follower
                        p_m_to_f = run_granger_test(diff_series[follower], diff_series[maestro_ch], lag=lag)
                        
                        # Direction 2: Follower -> Maestro
                        p_f_to_m = run_granger_test(diff_series[maestro_ch], diff_series[follower], lag=lag)
                        
                        records.append({
                            'treatment': treatment,
                            'plant_id': plant,
                            'feature': feat,
                            'lag': lag,
                            'follower_channel': follower,
                            'maestro_to_follower_p': p_m_to_f,
                            'follower_to_maestro_p': p_f_to_m,
                            'maestro_to_follower_sig': 1 if p_m_to_f < 0.05 else 0,
                            'follower_to_maestro_sig': 1 if p_f_to_m < 0.05 else 0
                        })

    # Save to CSV
    df_gc = pd.DataFrame(records)
    df_gc.to_csv('csv/granger_causality_results.csv', index=False)
    print("Granger Causality results saved to csv/granger_causality_results.csv")
    
    # 4. PLOTTING GRANGER CAUSALITY RESULTS (1x3 Subplots comparing Lags)
    plot_data = []
    for idx, row in df_gc.iterrows():
        plot_data.append({
            'Treatment': row['treatment'].replace('_', ' ').title(),
            'Feature': row['feature'],
            'Lag': f"Lag {row['lag']}",
            'Direction': 'Maestro → Followers',
            'Significant': row['maestro_to_follower_sig']
        })
        plot_data.append({
            'Treatment': row['treatment'].replace('_', ' ').title(),
            'Feature': row['feature'],
            'Lag': f"Lag {row['lag']}",
            'Direction': 'Followers → Maestro',
            'Significant': row['follower_to_maestro_sig']
        })
        
    df_plot = pd.DataFrame(plot_data)
    
    # Calculate percentage significant per group
    df_plot_grouped = df_plot.groupby(['Treatment', 'Feature', 'Lag', 'Direction'])['Significant'].mean().reset_index()
    df_plot_grouped['Significant (%)'] = df_plot_grouped['Significant'] * 100
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    sns.set_theme(style="whitegrid")
    
    for idx, lag in enumerate([1, 2, 3]):
        ax = axes[idx]
        df_lag = df_plot_grouped[df_plot_grouped['Lag'] == f"Lag {lag}"]
        
        sns.barplot(
            data=df_lag,
            x='Feature',
            y='Significant (%)',
            hue='Direction',
            ax=ax,
            palette={'Maestro → Followers': '#d62728', 'Followers → Maestro': '#7f7f7f'},
            edgecolor='black',
            alpha=0.85
        )
        
        ax.set_title(f"Granger Causality (Lag {lag})", fontsize=14, fontweight='bold')
        ax.set_xlabel("Complexity Feature", fontsize=12)
        if idx == 0:
            ax.set_ylabel("Significant Causal Links (%)", fontsize=12)
        else:
            ax.set_ylabel("")
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # Annotate bars
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.annotate(f"{height:.1f}%",
                            (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='bottom',
                            fontsize=9, color='black',
                            xytext=(0, 2),
                            textcoords='offset points',
                            fontweight='bold')
        
        # Only show legend on the last subplot to avoid clutter
        if idx == 2:
            ax.legend(title='Causal Direction', title_fontsize='11', fontsize='10', loc='upper right')
        else:
            ax.get_legend().remove()
            
    plt.suptitle("Granger Causality Directionality in Plant Complexity Features (Lags 1, 2, 3)", fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    plot_path = "img/granger_causality_directionality.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Granger causality multi-lag plot saved to {plot_path}")

if __name__ == '__main__':
    main()
