# Hypernetwork Modulation in Soy Plant Electrome (Glycine max)

This repository contains the modular Python pipeline used to analyze electrical dynamics (electrome) in soy plants (Glycine max) under saline (osmotic) and cut/fire (mechanical) stress. 

Our work adapts the mathematical framework of Hypernetworks and Higher-Order Phase (Hyperlocking) [1] to plant electrophysiology, demonstrating that plant modular coordination is governed by a modulated complexity hypernetwork topology.

---

## Scientific Context and Core Findings (Double Dissociation)

This project bridges theoretical network physics with plant electrophysiology:
1. **Direct Phase Coupling (Phase Locking):** Applying Hilbert transforms to raw, filtered electrical signals reveals no significant direct phase coupling (PLV and TLI remain at noise levels, p_FDR > 0.05). This is mathematically explained by a frequency mismatch modeled as a Wiener process (random walk), where phase difference variance grows linearly in time, pulling locking values towards the asymptotic noise limit 1/sqrt(T) = 0.0096.
2. **Complexity Modulation (Hypernetwork Topology):** In contrast, slow-moving complexity envelopes (Approximate Entropy, Detrended Fluctuation Analysis, and Power Spectral Density) exhibit a clear transition to a coordinated state post-stress. The unconditioned Mutual Information (MI) increases substantially, but collapses by 70% to 91% when conditioned on the designated maestro channel (Stem C1 for Salinity; Stressed Leaf C3 for Cut/Fire).
3. **Validation Proofs:**
   * **Surrogate Data Test:** Permuting (shuffling) the maestro series breaks the CMI collapse (the CMI remains high, matching unconditioned MI), confirming that the colapso is specific to the real temporal trajectory of the maestro.
   * **Label Permutation Null Model:** Conditioning the CMI on follower channels (C2 or C4) instead of the true maestro yields no information colapso, proving topological specificity.
   * **Transfer Entropy (Lags 1, 2, 3):** Establishes the non-linear active directionality of information flow (Maestro -> Followers) across different lags, confirming that the maestro actively drives the complexity of the followers.
   * **Stuart-Landau Control Simulation:** Simulating 3 phase oscillators driven by a modulator under Wiener noise replicates this exact behavior, proving the mathematical validity of our pipeline under high noise.

---

## Pipeline Architecture (script/)

The pipeline consists of modular, sequential Python 3 scripts:

1. **`01_extract_phase.py` (Phase Extraction):**
   * Filters raw voltage series (62.5 Hz) using a 5th-order Butterworth bandpass filter (0.05 - 2.0 Hz) to remove network aliasing.
   * Computes the analytical signal via the Hilbert Transform, extracts unwrapped continuous phase, and interpolates onto a 1 Hz aligned temporal grid relative to t_stress = 0.
2. **`02_pairwise_sync.py` (Pairwise PLV):**
   * Calculates continuous phase differences and pairwise PLVs.
   * Plots phase drift distributions showing continuous linear drift.
3. **`03_triplet_locking.py` (Triplet TLI):**
   * Computes triadic phase relations for Stem-modulates and Leaf-modulates hypotheses.
   * Generates continuous triplet phase drift plots aligned dynamically at t=0 using np.argmin(np.abs(t)).
4. **`04_conditional_info.py` (Conditional Mutual Information):**
   * Extracts complexity features (ApEn, DFA, PSD_mean) in 1-minute windows.
   * Computes pairwise MI and CMI conditioned on the real vs. shuffled maestro.
   * Generates 2x2 comparison heatmaps.
5. **`05_statistical_analysis.py` (Hypothesis Testing and FDR):**
   * Performs Shapiro-Wilk normality tests.
   * Conducts Wilcoxon signed-rank (paired) and Mann-Whitney U (independent) tests.
   * Applies Benjamini-Hochberg FDR correction and renders annotated boxplots.
6. **`06_stuart_landau_sim.py` (Stuart-Landau Sweep Control):**
   * Ground truth simulation showing how CMI colapso is sensitive to the modulation strength K while direct phase locking is masked by Wiener noise.
7. **`07_advanced_cmi_analysis.py` (Sliding Windows and Label Permutation):**
   * Estimates CMI in 15-minute sliding windows to capture rapid transitions.
   * Runs the label permutation null model, comparing CMI across all potential maestros.
8. **`08_transfer_entropy.py` (Causal Information Flow):**
   * Performs plant-by-plant Transfer Entropy testing at Lags 1, 2, and 3 on complexity features to establish directional routing.

---

## Summary of CMI Reduction (After Stress)

Conditioning the coordination structure on the true maestro yields a massive, specific drop in shared information:

| Treatment | Metric / Feature | Pairwise MI (bits) | CMI Real Maestro (bits) | CMI Shuffled Maestro (bits) | Information Reduction (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Corte e Fogo** | ApEn | 0.126 | 0.027 | 0.126 | **78.6%** |
| | DFA | 0.082 | 0.032 | 0.084 | **61.0%** |
| | PSD_mean | 0.080 | 0.019 | 0.086 | **76.3%** |
| **Salinidade** | ApEn | 0.251 | 0.068 | 0.276 | **72.9%** |
| | DFA | 0.211 | 0.019 | 0.208 | **91.0%** |
| | PSD_mean | 0.185 | 0.050 | 0.195 | **73.0%** |

---

## How to Run

### 1. Requirements
Install the required scientific packages:
```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn
```

### 2. Execution
Run the pipeline steps sequentially from the root directory:
```bash
# Step 1: Preprocess and extract phases
python3 script/01_extract_phase.py

# Step 2: Compute pairwise PLV and drift
python3 script/02_pairwise_sync.py

# Step 3: Compute triplet TLI and drift
python3 script/03_triplet_locking.py

# Step 4: Estimate CMI and Surrogate tests
python3 script/04_conditional_info.py

# Step 5: Inferential statistics and FDR
python3 script/05_statistical_analysis.py

# Step 6: Stuart-Landau sweep simulation (Ground Truth)
python3 script/06_stuart_landau_sim.py

# Step 7: 15-min Sliding window & Label permutation null model
python3 script/07_advanced_cmi_analysis.py

# Step 8: Multi-lag Transfer Entropy directionality
python3 script/08_transfer_entropy.py
```

All data output tables will be saved under csv/ and generated plots under img/.

---

## References

* **[1] Nijholt, E., Pereira, T., Wolfrum, M., et al.** (2026). *Hypernetworks induce stable hyperlocking*. Nature Communications, 17(1), 74556.
* **[2] Debono, M., et al.** (2025). *Abiotic stress triggers electrical synchronisation of shoot and leaves in soybean plants: a clue for plant attention-like*. (Preprint).
* **[3] Pincus, S.** (1995). *Approximate entropy (ApEn) as a complexity measure*. Chaos, 5(1), 110-117.
* **[4] Kraskov, A., Stögbauer, H., & Grassberger, P.** (2004). *Estimating mutual information*. Physical Review E, 69(6), 066138.
* **[5] Granger, C. W. J.** (1969). *Investigating causal relations by econometric models*. Econometrica, 37(3), 424-438.
* **[6] Benjamini, Y., & Hochberg, Y.** (1995). *Controlling the false discovery rate*. Journal of the Royal Statistical Society B, 57(1), 289-300.
