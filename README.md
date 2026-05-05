# CoinageClusterML

[![DOI](https://img.shields.io/badge/Dataset-Open%20QCD-1F4E79?style=flat-square)](http://muellergroup.jhu.edu/qcd)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Python ≥ 3.9](https://img.shields.io/badge/Python-≥3.9-blue.svg?style=flat-square)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Active-success.svg?style=flat-square)]()

> **Interpretable Machine-Learning Prediction of DFT Energies per Atom and Identification of Magic Numbers in Coinage-Metal Nanoclusters (N ≤ 55) from the Open Quantum Cluster Database**
>
> An end-to-end, reproducible ML pipeline for Cu, Ag, and Au nanoclusters built on the open Quantum Cluster Database (QCD). It benchmarks seven regressors with rigorous cross-validation, interprets predictions via SHAP, and identifies magic-number stability peaks separately for each coinage metal.

---

## Highlights

- **LightGBM** achieves test MAE = **0.0144 eV/atom** (R² = **0.996**) on a stratified 70/15/15 train/validation/test split of 4,381 Cu/Ag/Au clusters from the open QCD.
- **Geometry-only variant** (no DFT-derived electronic inputs) retains 97 % of full-model accuracy — usable directly from XYZ coordinates.
- **Per-metal Δ²E analysis** identifies universal magic numbers (N = 8, 34) and metal-specific peaks (N = 32, 38 for Au; relativistic stabilisation), and reveals an anomalous N = 20 in Au.
- **Two cross-validation protocols** — standard 5-fold and **size-grouped** (each fold leaves out a whole nuclearity bin) — quantitatively bound the model's interpolation/extrapolation regime.
- **SHAP** attribution shows that metal identity and cluster size N dominate predictions; geometric descriptors govern the small (~10 meV/atom) energy differences that determine magic-number locations.
- All figures, tables, trained models, and the dataset are released to support FAIR data practices.

---

## Table of Contents

- [Scientific Background](#scientific-background)
- [What's New in R1](#whats-new-in-r1)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Methodology](#methodology)
  - [1. Dataset](#1-dataset)
  - [2. Feature Engineering](#2-feature-engineering)
  - [3. Target Variable](#3-target-variable)
  - [4. Machine-Learning Models](#4-machine-learning-models)
  - [5. Validation Protocols](#5-validation-protocols)
  - [6. Interpretability](#6-interpretability)
  - [7. Magic-Number Identification](#7-magic-number-identification)
- [Key Results](#key-results)
- [Generated Outputs](#generated-outputs)
- [Reproducibility](#reproducibility)
- [Citation](#citation)
- [License](#license)
- [Authors](#authors)

---

## Scientific Background

Atomically precise coinage-metal nanoclusters (Cu, Ag, Au) exhibit pronounced size-dependent stability that is critical for catalysis, plasmonics, photocatalysis, and electronic devices. Direct first-principles screening becomes computationally prohibitive beyond ~55 atoms, motivating data-driven surrogates trained on existing DFT databases.

This repository implements an interpretable ML framework that learns the **DFT energy per atom (E_DFT/N)** within the QCD chemical and size domain, identifies **magic-number stability peaks per metal**, and quantifies its own scope through size-aware cross-validation.

---

## What's New in R1

This release accompanies the revised manuscript (PCCP, R1) and includes substantial improvements over the originally submitted version:

| Change | Why |
|---|---|
| **Cluster size N added as an explicit feature** | Was missing in the original code; restoration removes a confounding proxy effect on radius of gyration. |
| **Train/Validation/Test = 70/15/15 split** | Validation set now exists; replaces the previous 85/15. |
| **5-fold KFold + size-grouped cross-validation** | Provides statistical robustness and quantifies extrapolation cost across nuclearity bins. |
| **Best model is now LightGBM** (MAE = 0.0144 eV/atom) | Marginally better than ExtraTrees; both reported. |
| **New geometry-only model** | Same pipeline without HOMO–LUMO and magnetic moment — supports rapid screening from coordinates. |
| **Per-metal Δ²E analysis with adaptive thresholds** | Magic-number detection is now metal-specific; new physical insights for Au. |
| **Target renamed E_DFT/N** (not "binding energy") | Honest terminology; the quantity stored in QCD. |
| **Honest extrapolation discussion** | Size-grouped CV explicitly bounds the model's applicability domain. |

---

## Repository Structure

```
CoinageClusterML/
│
├── data/
│   └── CuAgAu.xlsx                        # QCD-derived Cu/Ag/Au structures (N ≤ 55)
│
├── src/
│   └── CoinageClusterML.py                # Full revised ML pipeline
│
├── notebooks/
│   └── coinageclusterml.ipynb             # Jupyter version (Kaggle-compatible)
│
├── results/
│   ├── Figures/                           # 13 main + supporting figures (PNG, 600 dpi)
│   ├── Tables/                            # Performance, magic numbers, errors (CSV)
│   └── Models/                            # Pickled trained estimators (optional)
│
├── manuscript/
│   ├── CoinageClusterML_R1.docx           # Revised manuscript (R1)
│   ├── CoinageClusterML_Supplementary_R1.docx
│   ├── Response_to_Reviewers.docx
│   └── Graphical_Abstract.png
│
├── README.md
└── LICENSE
```

---

## Quick Start

### Installation

Requires Python ≥ 3.9.

```bash
git clone https://github.com/alikhairbek/CoinageClusterML.git
cd CoinageClusterML

pip install numpy pandas matplotlib seaborn scikit-learn
pip install xgboost lightgbm catboost shap statsmodels openpyxl
```

### Running the pipeline

```bash
python src/CoinageClusterML.py
```

This will execute the full pipeline end-to-end:

1. Load and preprocess the QCD-derived dataset (auto-detects Kaggle vs local paths).
2. Compute geometric descriptors from atomic coordinates.
3. Train and benchmark seven regressors on both feature sets (full and geometry-only).
4. Run 5-fold KFold and size-grouped cross-validation.
5. Compute SHAP attribution for the best model.
6. Perform per-metal magic-number Δ²E analysis.
7. Generate all figures, tables, and a summary report.

Outputs are written to `REVISED_PCCP_R1/` (or `/kaggle/working/REVISED_PCCP_R1/` in Kaggle).

### Running on Kaggle

The script auto-detects the Kaggle environment and reads from `/kaggle/input/` if available. Add the `CuAgAu.xlsx` dataset to your Kaggle notebook's input, then run the script directly — no path edits needed.

---

## Methodology

### 1. Dataset

All structures and DFT energies are taken from the **open Quantum Cluster Database (QCD)** released by Manna *et al.* (Nature Sci. Data, 2023). The file `CuAgAu.xlsx` contains 4,381 unique Cu/Ag/Au cluster structures with N ≤ 55, computed at the **PBE-GGA + PAW** level of theory in plane-wave VASP. Each entry includes:

- atomic coordinates (relaxed)
- cluster size N and metal identity
- HOMO–LUMO gap
- magnetic moment
- valence-electron count
- total DFT energy

**Scope and limitations.** Because PBE typically overbinds metallic systems by ~0.1–0.2 eV/atom relative to hybrid functionals and underestimates HOMO–LUMO gaps, our ML predictions inherit this bias. Relative quantities such as Δ²E(N) are far less sensitive to functional choice and are therefore the basis of our magic-number analysis.

**Data sources:**
- Database: [Open Quantum Cluster Database](http://muellergroup.jhu.edu/qcd)
- Reference: [Manna *et al.*, *Sci Data* **10**, 308 (2023)](https://www.nature.com/articles/s41597-023-02200-4)

### 2. Feature Engineering

Two feature sets are constructed from each relaxed structure:

**Full feature set (14 inputs):**

| Group | Features |
|---|---|
| Geometric (9) | mean / std / max distance from centroid, radius of gyration *R*<sub>g</sub>, asphericity, bounding-box x/y/z, compactness *R*<sub>g</sub>/*d*<sub>mean</sub> |
| Composition (2) | metal Z, cluster size N |
| Electronic (3) | HOMO–LUMO gap, magnetic moment, valence-electron count |

**Geometry-only feature set (11 inputs):** the full set minus HOMO–LUMO gap and magnetic moment — i.e. only inputs computable from atomic coordinates without further DFT.

All features are standardised (zero mean, unit variance) using a scaler fitted on the training partition only.

### 3. Target Variable

The regression target is the **DFT energy per atom**:

```
E_DFT / N
```

This is **not** the standard atomization (binding) energy *E*<sub>at</sub>/N = *E*<sub>atom</sub> – *E*<sub>DFT</sub>/N, which would require an atomic-reference calculation per metal. We retain *E*<sub>DFT</sub>/N because:

1. It is the quantity directly stored in QCD.
2. Magic-number locations depend on the **second difference** Δ²E(N), which is invariant under any constant shift such as *E*<sub>atom</sub>.
3. The reported MAE values are also invariant under such a shift.

### 4. Machine-Learning Models

Seven regressors are benchmarked under identical conditions:

| Category | Models |
|---|---|
| Tree ensembles | ExtraTrees, RandomForest |
| Gradient boosting | XGBoost, LightGBM, CatBoost, GradientBoosting |
| Neural network | Multilayer Perceptron (MLP) |

Both the full and geometry-only feature sets are evaluated for every model.

### 5. Validation Protocols

Three complementary validation protocols are applied:

1. **Stratified 70/15/15 train / validation / test split** (preserves metal proportions across all subsets). The validation set is used solely for monitoring; no hyperparameter tuning is performed against the test set.
2. **5-fold KFold cross-validation** on the full dataset — within-domain stability estimate.
3. **Size-grouped cross-validation** with four nuclearity bins (N = 3–15, 16–30, 31–45, 46–55); each fold leaves one bin out — near-extrapolation stress test across size domains.

### 6. Interpretability

Model predictions are analysed via:

- **SHAP** (SHapley Additive exPlanations) attribution for the LightGBM full-feature model
- impurity-based global feature importance
- learning curves
- residual distributions
- size- and metal-stratified error analysis
- PCA / t-SNE projections of feature space

### 7. Magic-Number Identification

Stability peaks are identified using the second-difference index:

```
Δ²E(N) = E(N + 1) + E(N − 1) − 2 E(N)
```

Critically, the analysis is performed **separately for each metal** (Cu, Ag, Au) using **adaptive metal-specific thresholds**:

```
Δ²E_min = 0.5 × σ(Δ²E_metal)
```

yielding cut-offs of 0.0146, 0.0113, and 0.0112 eV/atom for Cu, Ag, and Au respectively. Sizes whose Δ²E exceeds the cut-off are reported as detected magic numbers and compared against literature compilations.

---

## Key Results

### Model performance (test set, N ≤ 55)

| Model | MAE (eV/atom) | R² | 5-fold CV MAE | Size-grouped CV MAE |
|---|---|---|---|---|
| **LightGBM** | **0.01436** | **0.9957** | 0.01422 ± 0.00044 | 0.107 ± 0.074 |
| ExtraTrees | 0.01453 | 0.9948 | 0.01389 ± 0.00027 | 0.079 ± 0.081 |
| CatBoost | 0.01534 | 0.9947 | 0.01463 ± 0.00039 | 0.079 ± 0.076 |
| RandomForest | 0.01548 | 0.9940 | 0.01507 ± 0.00030 | 0.092 ± 0.072 |
| GradientBoosting | 0.01550 | 0.9943 | 0.01400 ± 0.00061 | 0.086 ± 0.071 |
| XGBoost | 0.01597 | 0.9943 | 0.01457 ± 0.00027 | 0.142 ± 0.071 |
| NeuralNet (MLP) | 0.02908 | 0.9748 | 0.02550 ± 0.00112 | 0.078 ± 0.069 |

The order-of-magnitude gap between random 5-fold CV and size-grouped CV explicitly bounds the model's interpolation regime.

### Geometry-only vs full-feature comparison

| Model | Full MAE | Geo-only MAE | Increase |
|---|---|---|---|
| LightGBM | 0.01436 | 0.01477 | **+2.9 %** |
| ExtraTrees | 0.01453 | 0.01513 | +4.1 % |
| RandomForest | 0.01548 | 0.01585 | +2.4 % |

Removing the two electronic features costs only ~0.4 meV/atom for the best model, demonstrating that energies can be ranked accurately from coordinates alone within the QCD domain.

### Per-metal magic numbers

| Metal | Detected (Δ²E > 0.5σ) | Universal (with Cu, Ag, Au) | Metal-specific |
|---|---|---|---|
| Cu | {8, 10, 12, 14, 20, 24, 34, 37} | 8, 20, 34 | – |
| Ag | {6, 8, 10, 12, 20, 24, 26, 32, 34, 37, 53} | 6, 8, 20, 34 | – |
| Au | {6, 7, 8, 10, 12, 14, 16, 19, 25, 28, **32**, 34, 36, **38**, 53} | 6, 8, 34 | **N = 32, 38** (relativistic) |

**Au-specific findings:**
- Clear N = 32 and N = 38 peaks attributable to relativistic stabilisation of the 6s shell.
- Anomalous Δ²E = –0.002 eV/atom at N = 20 — in contrast to clear peaks for Cu and Ag at the same nuclearity. Plausibly attributable to the coexistence of Au₂₀ tetrahedral and lower-symmetry isomers reported in the literature.

---

## Generated Outputs

Running the pipeline produces:

### Figures (`results/Figures/`)

| Figure | Content |
|---|---|
| Fig 1 | Parity plot — LightGBM, full feature set |
| Fig 2 | Parity plot — geometry-only model |
| Fig 3 | Learning curve of best model |
| Fig 4–6 | Per-metal Δ²E analyses (Cu, Ag, Au) |
| Fig 7 | SHAP summary plot |
| Fig 8 | Feature-importance bar chart |
| Fig 9–10 | PCA / t-SNE projections |
| Fig 11 | Full vs geometry-only MAE comparison |
| Fig 12–13 | Per-sample error vs N; error distribution by metal |
| Fig S1–S11 | Geometric/electronic descriptor distributions, Rg-vs-N scaling, stability landscapes |

### Tables (`results/Tables/`)

| Table | Content |
|---|---|
| Table 1 | Model benchmarking (full features) with CV columns |
| Table 2 | ML predictions vs DFT, grouped by N |
| Table 3 | Per-metal magic numbers with literature comparison |
| Table 4 | Full vs geometry-only MAE comparison |

### Report

`REPORT.txt` — auto-generated text summary of all numerical results, including dataset scope, splits used, CV outcomes, and per-metal magic-number lists.

---

## Reproducibility

All randomness is controlled. The pipeline produces bit-identical outputs across runs given the same input file and Python environment.

| Component | Setting |
|---|---|
| Random seed | 42 (NumPy, scikit-learn, XGBoost, LightGBM, CatBoost) |
| Train / Val / Test split | Stratified 70/15/15 by metal |
| 5-fold KFold | shuffle = True, random_state = 42 |
| Size-grouped CV | 4 bins: N = 3–15, 16–30, 31–45, 46–55 |
| Magic-number threshold | 0.5 × σ(Δ²E) per metal |
| Standardisation | Fitted on training partition only |

### Tested software versions

```
Python 3.12     pandas 2.2     scikit-learn 1.4     XGBoost 2.0
LightGBM 4.3    CatBoost 1.2   SHAP 0.45            matplotlib 3.8
```

---

## Citation

If you use this code or framework in your work, please cite the manuscript:

```bibtex
@article{Khairbek2026CoinageClusterML,
  title   = {Interpretable Machine-Learning Prediction of DFT Energies per Atom
             and Identification of Magic Numbers in Coinage-Metal Nanoclusters
             (N $\leq$ 55) from the Open Quantum Cluster Database},
  author  = {Khairbek, Ali A. and Al-Zaben, Maha I. and
             Alzahrani, Abdullah Yahya Abdullah and Thomas, Renjith},
  journal = {Physical Chemistry Chemical Physics},
  year    = {2026},
  note    = {Submitted}
}
```

Please also cite the underlying QCD dataset:

```bibtex
@article{Manna2023QCD,
  author  = {Manna, S. and Wang, Y. and Hernandez, A. and Lile, P. and
             Liu, S. and Mueller, T.},
  title   = {A Database of Low-Energy Atomically Precise Nanoclusters},
  journal = {Scientific Data},
  volume  = {10},
  pages   = {308},
  year    = {2023},
  doi     = {10.1038/s41597-023-02200-4}
}
```

---

## License

This repository is released under the **MIT License**. See [LICENSE](LICENSE) for details.

The QCD reference data are distributed by Manna *et al.* under their own terms — please consult the [original repository](http://muellergroup.jhu.edu/qcd) for redistribution conditions.

---

## Authors

| | |
|---|---|
| **Ali A. Khairbek** ¹ | Centre for Theoretical and Computational Chemistry, St Berchmans College (Autonomous), Kerala, India |
| **Maha I. Al-Zaben** ² | Department of Chemistry, College of Science, King Saud University, Riyadh, Saudi Arabia |
| **Abdullah Yahya Abdullah Alzahrani** ³ | Department of Chemistry, Faculty of Science, King Khalid University, Abha, Saudi Arabia |
| **Renjith Thomas** ¹,⁴ ✉ | Department of Chemistry, St Berchmans College (Autonomous), Mahatma Gandhi University, Kerala, India |

**Corresponding authors:**
- Prof. Renjith Thomas — `renjith@sbcollege.ac.in`
- Dr. Ali A. Khairbek — `alikhairbek@gmail.com`

---

## Acknowledgements

We thank the developers of the open Quantum Cluster Database (Mueller group, Johns Hopkins University) for releasing this remarkable resource under FAIR principles, and the maintainers of scikit-learn, LightGBM, XGBoost, CatBoost, and SHAP for the open-source ML tooling that makes interpretable nanocluster modelling possible.
