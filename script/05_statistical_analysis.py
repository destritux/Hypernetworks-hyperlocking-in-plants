import os
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

def benjamini_hochberg_correct(p_values):
    p_vals = np.array(p_values)
    n = len(p_vals)
    sorted_indices = np.argsort(p_vals)
    sorted_p_vals = p_vals[sorted_indices]
    q_vals = np.zeros(n)
    prev_q = 1.0
    for i in range(n - 1, -1, -1):
        q = sorted_p_vals[i] * n / (i + 1)
        q = min(q, prev_q)
        q_vals[i] = q
        prev_q = q
    corrected_p_vals = np.zeros(n)
    corrected_p_vals[sorted_indices] = q_vals
    return corrected_p_vals

def main():
    print("Loading PLV and TLI results...")
    if not os.path.exists('csv/plv_results.csv') or not os.path.exists('csv/triplet_locking.csv'):
        raise FileNotFoundError("plv_results.csv or triplet_locking.csv not found! Run scripts 02 and 03 first.")
        
    plv_df = pd.read_csv('csv/plv_results.csv')
    tli_df = pd.read_csv('csv/triplet_locking.csv')
    
    # 1. Prepare paired datasets per plant
    # For PLV, we average across all 6 pairs per plant to get a single PLV index for baseline vs post-stress
    plv_paired = plv_df.groupby(['treatment', 'plant_id'])[['plv_before', 'plv_after']].mean().reset_index()
    
    # For TLI, we use the correct modulator hypothesis:
    # - cut_fire: stressed_leaf_modulates
    # - salt: stem_modulates
    tli_cf = tli_df[(tli_df['treatment'] == 'cut_fire') & (tli_df['hypothesis'] == 'stressed_leaf_modulates')].copy()
    tli_salt = tli_df[(tli_df['treatment'] == 'salt') & (tli_df['hypothesis'] == 'stem_modulates')].copy()
    tli_paired = pd.concat([tli_cf, tli_salt], ignore_index=True)
    
    # Merge PLV and TLI data
    merged_df = pd.merge(
        plv_paired, 
        tli_paired[['treatment', 'plant_id', 'hypothesis', 'tli_before', 'tli_after']], 
        on=['treatment', 'plant_id']
    )
    
    # Calculate differences (after - before)
    merged_df['plv_diff'] = merged_df['plv_after'] - merged_df['plv_before']
    merged_df['tli_diff'] = merged_df['tli_after'] - merged_df['tli_before']
    
    stats_records = []
    
    # 2. Shapiro-Wilk Normality Tests (Uncorrected, as normality testing is conservative here)
    for treatment in ['cut_fire', 'salt']:
        sub = merged_df[merged_df['treatment'] == treatment]
        
        for col in ['plv_before', 'plv_after', 'plv_diff', 'tli_before', 'tli_after', 'tli_diff']:
            shapiro_stat, shapiro_p = stats.shapiro(sub[col])
            stats_records.append({
                'analysis': 'normality_shapiro',
                'treatment': treatment,
                'variable': col,
                'statistic': shapiro_stat,
                'p_value_raw': shapiro_p,
                'p_value_corrected_fdr': shapiro_p, # not corrected for normality tests
                'interpretation': 'Normal' if shapiro_p > 0.05 else 'Non-Normal'
            })
            
    # 3. Hypothesis Tests
    hyp_tests = []
    
    # Paired tests (Before vs After within same plant)
    for treatment in ['cut_fire', 'salt']:
        sub = merged_df[merged_df['treatment'] == treatment]
        
        # Determine normality to select t-test or Wilcoxon
        norm_tli = next(r['interpretation'] == 'Normal' for r in stats_records if r['treatment'] == treatment and r['variable'] == 'tli_diff')
        norm_plv = next(r['interpretation'] == 'Normal' for r in stats_records if r['treatment'] == treatment and r['variable'] == 'plv_diff')
        
        # TLI Test
        if norm_tli:
            stat_tli, p_tli = stats.ttest_rel(sub['tli_after'], sub['tli_before'])
            test_name = 'Paired t-test'
        else:
            stat_tli, p_tli = stats.wilcoxon(sub['tli_after'], sub['tli_before'])
            test_name = 'Wilcoxon signed-rank'
            
        hyp_tests.append({
            'analysis': 'paired_difference_tli',
            'treatment': treatment,
            'variable': 'tli_after_vs_before',
            'test_type': test_name,
            'statistic': stat_tli,
            'p_value_raw': p_tli
        })
        
        # PLV Test
        if norm_plv:
            stat_plv, p_plv = stats.ttest_rel(sub['plv_after'], sub['plv_before'])
            test_name_plv = 'Paired t-test'
        else:
            stat_plv, p_plv = stats.wilcoxon(sub['plv_after'], sub['plv_before'])
            test_name_plv = 'Wilcoxon signed-rank'
            
        hyp_tests.append({
            'analysis': 'paired_difference_plv',
            'treatment': treatment,
            'variable': 'plv_after_vs_before',
            'test_type': test_name_plv,
            'statistic': stat_plv,
            'p_value_raw': p_plv
        })
        
    # Independent Test (Cut/Fire vs Salt on TLI change)
    cf_sub = merged_df[merged_df['treatment'] == 'cut_fire']
    salt_sub = merged_df[merged_df['treatment'] == 'salt']
    
    norm_cf = next(r['interpretation'] == 'Normal' for r in stats_records if r['treatment'] == 'cut_fire' and r['variable'] == 'tli_diff')
    norm_salt = next(r['interpretation'] == 'Normal' for r in stats_records if r['treatment'] == 'salt' and r['variable'] == 'tli_diff')
    
    if norm_cf and norm_salt:
        stat_comp, p_comp = stats.ttest_ind(cf_sub['tli_diff'], salt_sub['tli_diff'])
        test_comp_name = 'Independent t-test'
    else:
        stat_comp, p_comp = stats.mannwhitneyu(cf_sub['tli_diff'], salt_sub['tli_diff'])
        test_comp_name = 'Mann-Whitney U'
        
    hyp_tests.append({
        'analysis': 'treatment_comparison_tli_diff',
        'treatment': 'cut_fire_vs_salt',
        'variable': 'tli_diff',
        'test_type': test_comp_name,
        'statistic': stat_comp,
        'p_value_raw': p_comp
    })
    
    # Apply Benjamini-Hochberg FDR correction on the 5 hypothesis tests
    raw_p_vals = [t['p_value_raw'] for t in hyp_tests]
    corrected_p_vals = benjamini_hochberg_correct(raw_p_vals)
    
    for idx, t in enumerate(hyp_tests):
        t['p_value_corrected_fdr'] = corrected_p_vals[idx]
        p_corr = corrected_p_vals[idx]
        t['interpretation'] = f"Significant (p_fdr={p_corr:.4f})" if p_corr < 0.05 else f"Not Significant (p_fdr={p_corr:.4f})"
        stats_records.append(t)
        
    # Save statistics report
    stats_df = pd.DataFrame(stats_records)
    stats_df.to_csv('csv/analise_estatistica_final.csv', index=False)
    print("Statistical report saved to csv/analise_estatistica_final.csv")
    print(stats_df)
    
    # 5. Visualizing the distributions (Boxplots)
    plt.figure(figsize=(14, 6))
    
    plot_records = []
    for idx, row in merged_df.iterrows():
        plot_records.append({
            'Treatment': row['treatment'].replace('_', ' ').title(),
            'Condition': 'Before',
            'Metric': 'Pairwise PLV',
            'Value': row['plv_before']
        })
        plot_records.append({
            'Treatment': row['treatment'].replace('_', ' ').title(),
            'Condition': 'After',
            'Metric': 'Pairwise PLV',
            'Value': row['plv_after']
        })
        plot_records.append({
            'Treatment': row['treatment'].replace('_', ' ').title(),
            'Condition': 'Before',
            'Metric': 'Triplet TLI',
            'Value': row['tli_before']
        })
        plot_records.append({
            'Treatment': row['treatment'].replace('_', ' ').title(),
            'Condition': 'After',
            'Metric': 'Triplet TLI',
            'Value': row['tli_after']
        })
        
    plot_df = pd.DataFrame(plot_records)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    
    treatments = ['Cut Fire', 'Salt']
    
    for idx, treat in enumerate(treatments):
        ax = axes[idx]
        sub_plot_df = plot_df[plot_df['Treatment'] == treat]
        
        sns.boxplot(
            data=sub_plot_df, 
            x='Metric', 
            y='Value', 
            hue='Condition', 
            ax=ax, 
            palette={'Before': '#1f77b4', 'After': '#d62728'},
            width=0.6
        )
        sns.stripplot(
            data=sub_plot_df, 
            x='Metric', 
            y='Value', 
            hue='Condition', 
            ax=ax, 
            dodge=True, 
            color='black', 
            alpha=0.5, 
            size=6,
            legend=False
        )
        
        ax.set_title(f"Locking Metrics: {treat}", fontsize=15, fontweight='bold')
        ax.set_xlabel("Locking Scale", fontsize=13)
        ax.set_ylabel("Locking Value", fontsize=13)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # Get raw and corrected p-values
        p_plv_fdr = next(r['p_value_corrected_fdr'] for r in stats_records if r['treatment'] == treat.lower().replace(' ', '_') and r['analysis'] == 'paired_difference_plv')
        p_tli_fdr = next(r['p_value_corrected_fdr'] for r in stats_records if r['treatment'] == treat.lower().replace(' ', '_') and r['analysis'] == 'paired_difference_tli')
        
        def get_sig_label(p):
            if p < 0.001: return '***'
            elif p < 0.01: return '**'
            elif p < 0.05: return '*'
            else: return 'ns'
            
        # Draw significance lines
        y_max_plv = sub_plot_df[sub_plot_df['Metric'] == 'Pairwise PLV']['Value'].max()
        ax.plot([0 - 0.15, 0 + 0.15], [y_max_plv * 1.05, y_max_plv * 1.05], color='black', lw=1.2)
        ax.text(0, y_max_plv * 1.07, f"{get_sig_label(p_plv_fdr)} (FDR)", ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        y_max_tli = sub_plot_df[sub_plot_df['Metric'] == 'Triplet TLI']['Value'].max()
        ax.plot([1 - 0.15, 1 + 0.15], [y_max_tli * 1.05, y_max_tli * 1.05], color='black', lw=1.2)
        ax.text(1, y_max_tli * 1.07, f"{get_sig_label(p_tli_fdr)} (FDR)", ha='center', va='bottom', fontsize=10, fontweight='bold')
        
    plt.suptitle("Phase Locking Comparison: Pairwise vs. Triplet (Hyperlocking)", fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plot_path = "img/statistical_comparisons.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to {plot_path}")

if __name__ == '__main__':
    main()
