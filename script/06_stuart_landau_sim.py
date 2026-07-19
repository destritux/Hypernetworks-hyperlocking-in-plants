import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression

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

def simulate_system(N, dt, K_factor, sigma=0.5):
    t = np.arange(N) * dt
    
    # Independent Wiener noise
    W1 = np.cumsum(np.random.normal(0, np.sqrt(dt), N)) * sigma
    W2 = np.cumsum(np.random.normal(0, np.sqrt(dt), N)) * sigma
    W3 = np.cumsum(np.random.normal(0, np.sqrt(dt), N)) * sigma
    
    # Modulator phase
    w3 = 0.2
    theta3 = w3 * t + W3
    
    # Coupling scaled by K_factor
    K1 = 1.5 * K_factor
    K2 = 1.2 * K_factor
    w1, w2 = 0.5, 0.4
    
    theta1 = w1 * t + K1 * np.sin(theta3) + W1
    theta2 = w2 * t + K2 * np.sin(theta3) + W2
    
    # Smooth derivative to get instantaneous frequency envelopes
    win = 100
    df1 = pd.Series(np.diff(theta1) / dt).rolling(win, min_periods=1).mean().values
    df2 = pd.Series(np.diff(theta2) / dt).rolling(win, min_periods=1).mean().values
    df3 = pd.Series(np.diff(theta3) / dt).rolling(win, min_periods=1).mean().values
    
    F1 = df1[win:-win]
    F2 = df2[win:-win]
    F3 = df3[win:-win]
    
    return theta1, theta2, theta3, F1, F2, F3, t

def main():
    print("Running Stuart-Landau Simulation with Coupling Parameter Sweep...")
    np.random.seed(42)
    
    N = 10000
    dt = 0.1
    
    # Sweep coupling factor K_factor from 0.0 to 2.0
    k_factors = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    
    mi_unc_list = []
    cmi_real_list = []
    cmi_shuff_list = []
    
    # For phase drift plot, we simulate at a representative K_factor = 1.5
    theta1_rep, theta2_rep, theta3_rep, _, _, _, t_rep = simulate_system(N, dt, K_factor=1.5)
    diff_rep = theta1_rep - theta2_rep
    plv_rep = np.abs(np.mean(np.exp(1j * diff_rep)))
    theta_tri_rep = theta1_rep + theta2_rep - 2 * theta3_rep
    tli_rep = np.abs(np.mean(np.exp(1j * theta_tri_rep)))
    
    print("  Sweeping coupling parameter K...")
    for k_f in k_factors:
        _, _, _, F1, F2, F3, _ = simulate_system(N, dt, K_factor=k_f)
        F3_shuffled = np.random.permutation(F3)
        
        mi_unc = pairwise_mutual_information(F1, F2)
        cmi_real = conditional_mutual_information(F1, F2, F3)
        cmi_shuff = conditional_mutual_information(F1, F2, F3_shuffled)
        
        print(f"    K_factor={k_f:.1f}: MI={mi_unc:.4f}, CMI_real={cmi_real:.4f}, CMI_shuff={cmi_shuff:.4f}")
        mi_unc_list.append(mi_unc)
        cmi_real_list.append(cmi_real)
        cmi_shuff_list.append(cmi_shuff)
        
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Phase Drift (Representative)
    axes[0].plot(t_rep, diff_rep - diff_rep[0], label=f"Pairwise Difference (PLV={plv_rep:.3f})", color='#1f77b4', alpha=0.8)
    axes[0].plot(t_rep, theta_tri_rep - theta_tri_rep[0], label=f"Triplet Phase (TLI={tli_rep:.3f})", color='#d62728', alpha=0.8)
    axes[0].set_title("Ground Truth Phase Drift (Wiener Noise, K=1.5)", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Time (s)", fontsize=11)
    axes[0].set_ylabel("Accumulated Phase Difference (rad)", fontsize=11)
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend(fontsize=10)
    
    # Right: MI and CMI as function of K_factor
    axes[1].plot(k_factors, mi_unc_list, marker='o', label="Pairwise MI I(F1; F2)", color='#2ca02c', lw=2.5)
    axes[1].plot(k_factors, cmi_shuff_list, marker='^', label="CMI Shuffled Maestro I(F1; F2 | F3_shuff)", color='#1f77b4', lw=2.5, ls='--')
    axes[1].plot(k_factors, cmi_real_list, marker='s', label="CMI Real Maestro I(F1; F2 | F3)", color='#d62728', lw=2.5)
    
    axes[1].set_title("Sincronização de Complexidade vs. Força de Modulação (K)", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Coupling Scale Factor (K)", fontsize=11)
    axes[1].set_ylabel("Shared Information (bits)", fontsize=11)
    axes[1].set_xticks(k_factors)
    axes[1].set_ylim(0, max(mi_unc_list + cmi_shuff_list) * 1.2)
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend(fontsize=10)
    
    plt.suptitle("Stuart-Landau Phase Dynamics: Direct Phase Drift vs. Complexity Modulation Sensitivity", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    os.makedirs('img', exist_ok=True)
    plot_path = "img/stuart_landau_simulation.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Simulation plot saved to {plot_path}")
    
    # Save simulated values to CSV for reference
    sim_df = pd.DataFrame({
        'k_factor': k_factors,
        'mi_unconditioned': mi_unc_list,
        'mi_conditioned_real': cmi_real_list,
        'mi_conditioned_shuffled': cmi_shuff_list
    })
    sim_df.to_csv('csv/stuart_landau_sim_results.csv', index=False)
    print("Simulation sweep results saved to csv/stuart_landau_sim_results.csv")

if __name__ == '__main__':
    main()
