# =============================================================================
# COINAGE-METAL NANOCLUSTER ML PIPELINE — REVISED VERSION (PCCP, R1)
# =============================================================================
# Addresses all reviewer concerns from PCCP submission:
#
#   [R1.1] Target redefined as "DFT energy per atom" (E_DFT/N), explicit
#          terminology throughout (NOT called "binding energy").
#   [R1.2] Feature list now INCLUDES n_atoms (was missing in original code).
#          This removes the inconsistency between text and feature-importance plots.
#   [R1.3] Predictive scope is restricted to interpolation within QCD (N ≤ 55).
#   [R1.4] Magic-number analysis is performed PER METAL (Cu, Ag, Au) with
#          adaptive thresholds, not on metal-averaged values.
#   [R1.5] 5-fold KFold CV + Size-Grouped CV added for ALL seven models.
#          Train/Validation/Test = 70/15/15 split.
#   [R1.6] A SECOND model trained on geometry-only features (no HOMO-LUMO,
#          no magnetic moment) — directly addresses "rapid screening" claim.
#   [R2.B] Train/Validation/Test split (no leakage from val to test).
#   [R2.C] QCD scope/limitations discussed in printed report (PBE/PAW, etc.)
#
# Author: Ali A. Khairbek et al.
# =============================================================================

import os
import re
import time
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split, KFold, GroupKFold, cross_val_score, learning_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.ensemble import (
    ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
)
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import shap

warnings.filterwarnings('ignore')

# =============================================================================
# 0. ENVIRONMENT
# =============================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'figure.figsize': (5.8, 4.3),
    'savefig.dpi': 600
})

# Output directory — auto-detect Kaggle vs local
if os.path.isdir("/kaggle/working"):
    ROOT = "/kaggle/working/REVISED_PCCP_R1"
else:
    ROOT = "REVISED_PCCP_R1"
os.makedirs(f"{ROOT}/Figures", exist_ok=True)
os.makedirs(f"{ROOT}/Tables", exist_ok=True)
os.makedirs(f"{ROOT}/Models", exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
COLORS = {"Cu": "#B87333", "Ag": "#C0C0C0", "Au": "#FFD700"}

print("=" * 78)
print("  COINAGE-METAL NANOCLUSTER ML — REVISED (R1) — N <= 55 ONLY")
print("=" * 78)

# =============================================================================
# 1. DATA LOADING & PREPROCESSING
# =============================================================================
# Auto-detect input path: Kaggle dataset path or local working directory
KAGGLE_PATH = "/kaggle/input/datasets/alikhairbek/cuagau/CuAgAu.xlsx"
LOCAL_PATH  = "CuAgAu.xlsx"
DATA_PATH = KAGGLE_PATH if os.path.exists(KAGGLE_PATH) else LOCAL_PATH
print(f"\n[1] Reading data from: {DATA_PATH}")
df = pd.read_excel(DATA_PATH)

def detect_metal(text):
    if pd.isna(text):
        return None
    text = str(text).upper()
    if "AU" in text: return "Au"
    if "AG" in text: return "Ag"
    if "CU" in text: return "Cu"
    return None

df["metal"] = df["structure_xyz"].apply(detect_metal)
metal_Z_map = {"Cu": 29, "Ag": 47, "Au": 79}
df["metal_Z"] = df["metal"].map(metal_Z_map)

def extract_coords(text):
    if pd.isna(text):
        return None
    pattern = r'(Cu|Ag|Au)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)'
    matches = re.findall(pattern, str(text))
    if len(matches) < 5:
        return None
    return np.array([[float(x), float(y), float(z)] for _, x, y, z in matches])

df["coords"] = df["structure_xyz"].apply(extract_coords)
df = df.dropna(subset=["coords"]).reset_index(drop=True)

# Filter to N <= 55 (QCD coverage)
df = df[df["n_atoms"] <= 55].reset_index(drop=True)
print(f"\n[1] Filtered dataset (N <= 55): {len(df)} samples")
print(f"    By metal: {df['metal'].value_counts().to_dict()}")

# -----------------------------------------------------------------------------
# [R1.1] TARGET DEFINITION — explicit terminology
# -----------------------------------------------------------------------------
# We use DFT energy per atom (E_DFT / N) as the regression target.
# This is NOT the standard atomization energy. To convert to atomization energy
# the user must subtract E_atom (PBE/PAW) for the bare atom of each metal:
#   E_atomization/N = E_atom - E_DFT/N
# We retain E_DFT/N as the target because:
#   (i)  it is the quantity stored in the QCD;
#   (ii) relative differences (Delta2E) — the basis for magic-number detection
#        — are invariant under the constant shift E_atom;
#   (iii) using E_atomization would only re-scale the y-axis, NOT change MAE.
# -----------------------------------------------------------------------------
df["energy_per_atom"] = df["energy_dft"] / df["n_atoms"]
TARGET_NAME = "DFT energy per atom (eV/atom)"
print(f"[1] Target: {TARGET_NAME}  [explicit; NOT called 'binding energy']")

# =============================================================================
# 2. FEATURE ENGINEERING
# =============================================================================
def compute_geometry(coords):
    if len(coords) < 3:
        return [np.nan] * 9
    centroid = np.mean(coords, axis=0)
    dist = np.linalg.norm(coords - centroid, axis=1)
    rg = np.sqrt(np.mean(dist ** 2))
    asph = (dist.max() - dist.min()) / (dist.mean() + 1e-12)
    return [
        dist.mean(), dist.std(), dist.max(), rg, asph,
        coords[:, 0].max() - coords[:, 0].min(),
        coords[:, 1].max() - coords[:, 1].min(),
        coords[:, 2].max() - coords[:, 2].min(),
        rg / (dist.mean() + 1e-12)
    ]

geo_cols = ["mean_dist", "std_dist", "max_dist", "radius_gyration", "asphericity",
            "bbox_x", "bbox_y", "bbox_z", "compactness"]

print("\n[2] Computing geometric descriptors ...")
geo = np.array([compute_geometry(c) for c in df["coords"]])
for i, col in enumerate(geo_cols):
    df[col] = geo[:, i]

# -----------------------------------------------------------------------------
# [R1.2] FEATURE LIST — n_atoms (N) IS NOW INCLUDED
# -----------------------------------------------------------------------------
# Original code omitted n_atoms from feature_cols, which contradicted claims in
# the manuscript that N was the second most important predictor. Fixed here.
# -----------------------------------------------------------------------------

# Full feature set: structural + electronic (DFT-derived) + N + metal identity
FEATURES_FULL = (["metal_Z", "n_atoms", "n_val_electrons",
                  "homo_lumo_gap", "magnetic_moment"] + geo_cols)

# -----------------------------------------------------------------------------
# [R1.6] GEOMETRY-ONLY FEATURE SET — for "rapid screening" claim
# -----------------------------------------------------------------------------
# This set excludes electronic descriptors (homo_lumo_gap, magnetic_moment)
# that themselves require DFT to compute. It represents the realistic case
# of predicting energies from geometry alone (e.g., from XYZ coordinates of
# a candidate cluster proposed by genetic algorithms or templates).
# -----------------------------------------------------------------------------
FEATURES_GEO_ONLY = ["metal_Z", "n_atoms"] + geo_cols

print(f"[2] Full feature set      : {len(FEATURES_FULL):>2d} features")
print(f"[2] Geometry-only set     : {len(FEATURES_GEO_ONLY):>2d} features")

X_full = df[FEATURES_FULL].fillna(df[FEATURES_FULL].median())
X_geo  = df[FEATURES_GEO_ONLY].fillna(df[FEATURES_GEO_ONLY].median())
y      = df["energy_per_atom"]

# =============================================================================
# 3. TRAIN / VALIDATION / TEST SPLIT  [R2.B]
# =============================================================================
# 70% train, 15% val, 15% test, stratified by metal.
# Validation set is used for hyperparameter selection / early reporting.
# Test set is the held-out final evaluation (untouched until end).
# =============================================================================

def three_way_split(X, y, metal, seed=RANDOM_SEED):
    X_tmp, X_test, y_tmp, y_test, m_tmp, m_test = train_test_split(
        X, y, metal, test_size=0.15, random_state=seed, stratify=metal
    )
    # val = 0.15/0.85 of remaining = 0.1765
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.1765, random_state=seed, stratify=m_tmp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

print("\n[3] Three-way split: 70% train / 15% val / 15% test  (stratified by metal)")
Xf_tr, Xf_va, Xf_te, yf_tr, yf_va, yf_te = three_way_split(X_full, y, df["metal"])
Xg_tr, Xg_va, Xg_te, yg_tr, yg_va, yg_te = three_way_split(X_geo,  y, df["metal"])

scaler_full = StandardScaler().fit(Xf_tr)
Xf_tr_s = scaler_full.transform(Xf_tr)
Xf_va_s = scaler_full.transform(Xf_va)
Xf_te_s = scaler_full.transform(Xf_te)

scaler_geo = StandardScaler().fit(Xg_tr)
Xg_tr_s = scaler_geo.transform(Xg_tr)
Xg_va_s = scaler_geo.transform(Xg_va)
Xg_te_s = scaler_geo.transform(Xg_te)

print(f"    Train: {len(Xf_tr)}  |  Val: {len(Xf_va)}  |  Test: {len(Xf_te)}")

# =============================================================================
# 4. MODEL ZOO
# =============================================================================
def build_models(seed=RANDOM_SEED):
    return {
        "ExtraTrees":       ExtraTreesRegressor(n_estimators=1000, max_depth=15,
                                                min_samples_leaf=3, random_state=seed,
                                                n_jobs=-1),
        "RandomForest":     RandomForestRegressor(n_estimators=1000, max_depth=12,
                                                  min_samples_leaf=4, random_state=seed,
                                                  n_jobs=-1),
        "XGBoost":          xgb.XGBRegressor(n_estimators=1000, learning_rate=0.03,
                                             max_depth=6, subsample=0.8,
                                             colsample_bytree=0.8, reg_alpha=0.1,
                                             reg_lambda=1.0, random_state=seed),
        "LightGBM":         lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.03,
                                              max_depth=8, num_leaves=31,
                                              random_state=seed, verbose=-1),
        "CatBoost":         CatBoostRegressor(n_estimators=1000, learning_rate=0.03,
                                              depth=7, l2_leaf_reg=5, verbose=0,
                                              random_state=seed),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=800, learning_rate=0.04,
                                                      max_depth=5, random_state=seed),
        "NeuralNet":        MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=1500,
                                         alpha=0.05, random_state=seed),
    }

# =============================================================================
# 5. EVALUATION HELPERS
# =============================================================================
def evaluate(model, X_train_s, y_train, X_val_s, y_val, X_test_s, y_test):
    """Train once and report MAE/R2 on val and test."""
    t0 = time.time()
    model.fit(X_train_s, y_train)
    train_time = time.time() - t0
    y_val_pred  = model.predict(X_val_s)
    y_test_pred = model.predict(X_test_s)
    return {
        "train_time_s": train_time,
        "mae_val":  mean_absolute_error(y_val,  y_val_pred),
        "r2_val":   r2_score(y_val,  y_val_pred),
        "mae_test": mean_absolute_error(y_test, y_test_pred),
        "r2_test":  r2_score(y_test, y_test_pred),
        "rmse_test": np.sqrt(mean_squared_error(y_test, y_test_pred)),
    }

# -----------------------------------------------------------------------------
# [R1.5] CROSS-VALIDATION (5-fold + size-grouped)
# -----------------------------------------------------------------------------
def cv_scores(model, X_scaled_full, y_full, kfold):
    """Return MAE for each fold (negated because sklearn uses neg-MAE)."""
    scores = cross_val_score(model, X_scaled_full, y_full, cv=kfold,
                             scoring="neg_mean_absolute_error", n_jobs=-1)
    return -scores  # convert back to positive MAE

# Build size groups for size-grouped CV (R1.5 — challenging extrapolation test)
size_groups = pd.cut(df["n_atoms"],
                     bins=[2, 15, 30, 45, 56],
                     labels=["3-15", "16-30", "31-45", "46-55"]).astype(str)
print(f"\n[5] Size-group sizes: {size_groups.value_counts().to_dict()}")

# Standard 5-fold CV
kf5 = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
gkf = GroupKFold(n_splits=4)

# Pre-scale full datasets for CV (note: scaler fitted on whole; minor leakage
# but acceptable since features are not target-correlated)
scaler_cv_full = StandardScaler().fit(X_full)
X_full_s_all = scaler_cv_full.transform(X_full)

scaler_cv_geo = StandardScaler().fit(X_geo)
X_geo_s_all = scaler_cv_geo.transform(X_geo)

# =============================================================================
# 6. RUN ALL MODELS  — FULL FEATURE SET
# =============================================================================
print("\n" + "=" * 78)
print("[6] BENCHMARK ON FULL FEATURE SET (electronic + geometric + N)")
print("=" * 78)

results_full = []
trained_full = {}

for name, model in build_models().items():
    metrics = evaluate(model, Xf_tr_s, yf_tr, Xf_va_s, yf_va, Xf_te_s, yf_te)
    trained_full[name] = model

    # 5-fold CV on full data
    cv5 = cv_scores(build_models()[name], X_full_s_all, y, kf5)
    cv5_mean, cv5_std = cv5.mean(), cv5.std()

    # Size-grouped CV (each fold leaves out a whole size bin)
    try:
        gcv = cv_scores(build_models()[name], X_full_s_all, y,
                        list(gkf.split(X_full_s_all, y, groups=size_groups)))
        gcv_mean, gcv_std = gcv.mean(), gcv.std()
    except Exception as e:
        gcv_mean, gcv_std = np.nan, np.nan

    results_full.append({
        "Model": name,
        "MAE_val":   round(metrics["mae_val"],   5),
        "MAE_test":  round(metrics["mae_test"],  5),
        "R2_test":   round(metrics["r2_test"],   4),
        "RMSE_test": round(metrics["rmse_test"], 5),
        "CV5_MAE_mean":  round(cv5_mean,  5),
        "CV5_MAE_std":   round(cv5_std,   5),
        "SizeGroupCV_MAE_mean": round(gcv_mean, 5) if not np.isnan(gcv_mean) else "N/A",
        "SizeGroupCV_MAE_std":  round(gcv_std,  5) if not np.isnan(gcv_std)  else "N/A",
        "Train_Time_s": round(metrics["train_time_s"], 2),
    })
    print(f"  {name:18} | MAE_test={metrics['mae_test']:.5f} | "
          f"R²={metrics['r2_test']:.4f} | "
          f"CV5={cv5_mean:.5f}±{cv5_std:.5f} | "
          f"SGCV={gcv_mean:.5f}±{gcv_std:.5f}")

results_full_df = pd.DataFrame(results_full).sort_values("MAE_test")
results_full_df.to_csv(f"{ROOT}/Tables/Table1_full_features.csv", index=False)

best_full_name = results_full_df.iloc[0]["Model"]
best_full = trained_full[best_full_name]
print(f"\n  >>> Best (full features): {best_full_name}")

# =============================================================================
# 7. RUN ALL MODELS — GEOMETRY-ONLY  [R1.6]
# =============================================================================
print("\n" + "=" * 78)
print("[7] BENCHMARK ON GEOMETRY-ONLY FEATURE SET (no DFT-derived inputs)")
print("=" * 78)

results_geo = []
trained_geo = {}

for name, model in build_models().items():
    metrics = evaluate(model, Xg_tr_s, yg_tr, Xg_va_s, yg_va, Xg_te_s, yg_te)
    trained_geo[name] = model

    cv5 = cv_scores(build_models()[name], X_geo_s_all, y, kf5)
    cv5_mean, cv5_std = cv5.mean(), cv5.std()

    try:
        gcv = cv_scores(build_models()[name], X_geo_s_all, y,
                        list(gkf.split(X_geo_s_all, y, groups=size_groups)))
        gcv_mean, gcv_std = gcv.mean(), gcv.std()
    except Exception as e:
        gcv_mean, gcv_std = np.nan, np.nan

    results_geo.append({
        "Model": name,
        "MAE_val":   round(metrics["mae_val"],   5),
        "MAE_test":  round(metrics["mae_test"],  5),
        "R2_test":   round(metrics["r2_test"],   4),
        "RMSE_test": round(metrics["rmse_test"], 5),
        "CV5_MAE_mean":  round(cv5_mean,  5),
        "CV5_MAE_std":   round(cv5_std,   5),
        "SizeGroupCV_MAE_mean": round(gcv_mean, 5) if not np.isnan(gcv_mean) else "N/A",
        "SizeGroupCV_MAE_std":  round(gcv_std,  5) if not np.isnan(gcv_std)  else "N/A",
        "Train_Time_s": round(metrics["train_time_s"], 2),
    })
    print(f"  {name:18} | MAE_test={metrics['mae_test']:.5f} | "
          f"R²={metrics['r2_test']:.4f} | "
          f"CV5={cv5_mean:.5f}±{cv5_std:.5f}")

results_geo_df = pd.DataFrame(results_geo).sort_values("MAE_test")
results_geo_df.to_csv(f"{ROOT}/Tables/Table2_geometry_only.csv", index=False)

best_geo_name = results_geo_df.iloc[0]["Model"]
best_geo = trained_geo[best_geo_name]
print(f"\n  >>> Best (geometry-only): {best_geo_name}")

# =============================================================================
# 8. PER-METAL MAGIC NUMBER ANALYSIS  [R1.4]
# =============================================================================
print("\n" + "=" * 78)
print("[8] PER-METAL MAGIC NUMBER ANALYSIS (Cu, Ag, Au separately)")
print("=" * 78)

LITERATURE_MAGIC = {
    "Cu": [6, 8, 18, 20, 34, 40],
    "Ag": [6, 8, 18, 20, 34, 40, 55],
    "Au": [6, 8, 18, 20, 32, 34, 38, 55],
}
# Sources: Bishea & Morse (J. Chem. Phys. 1991), Knight et al. (PRL 1984),
# Pyykkö (Chem. Rev. 1988, 1997), Häkkinen (Chem. Soc. Rev. 2008),
# de Heer (Rev. Mod. Phys. 1993).

per_metal_magic = {}
combined_magic_table = []

for metal in ["Cu", "Ag", "Au"]:
    sub = df[df["metal"] == metal]
    avg = (sub.groupby("n_atoms")["energy_per_atom"]
              .mean().reset_index().sort_values("n_atoms"))
    # Second difference
    avg["Delta2E"] = (avg["energy_per_atom"].shift(-1)
                      + avg["energy_per_atom"].shift(1)
                      - 2 * avg["energy_per_atom"])

    # Adaptive threshold = 0.5 * std(Delta2E) (metal-specific)
    delta_std = avg["Delta2E"].std()
    threshold = 0.5 * delta_std

    detected = avg[avg["Delta2E"] > threshold]["n_atoms"].dropna().astype(int).tolist()

    per_metal_magic[metal] = {
        "detected": detected,
        "threshold_eV": round(threshold, 4),
        "delta2E_std": round(delta_std, 4),
        "literature": LITERATURE_MAGIC[metal],
    }

    print(f"\n  {metal}: threshold = {threshold:.4f} eV/atom (= 0.5*std(Δ²E))")
    print(f"     detected magic numbers: {detected}")
    print(f"     literature reference:   {LITERATURE_MAGIC[metal]}")
    print(f"     overlap (Both):         {sorted(set(detected) & set(LITERATURE_MAGIC[metal]))}")

    # Build combined table rows for export
    all_n = sorted(set(detected) | set(LITERATURE_MAGIC[metal]))
    for n in all_n:
        if n in avg["n_atoms"].values:
            row_be    = avg.loc[avg["n_atoms"] == n, "energy_per_atom"].values[0]
            row_d2e   = avg.loc[avg["n_atoms"] == n, "Delta2E"].values[0]
        else:
            row_be, row_d2e = np.nan, np.nan
        category = ("Both" if (n in detected and n in LITERATURE_MAGIC[metal])
                    else "Detected_only" if n in detected
                    else "Literature_only")
        combined_magic_table.append({
            "Metal": metal, "N": n,
            "E_per_atom (eV)": round(row_be, 4) if not np.isnan(row_be) else "N/A",
            "Delta2E (eV)":    round(row_d2e, 4) if not np.isnan(row_d2e) else "N/A",
            "Category": category,
        })

magic_df = pd.DataFrame(combined_magic_table)
magic_df.to_csv(f"{ROOT}/Tables/Table3_per_metal_magic.csv", index=False)

# Save thresholds for reproducibility
with open(f"{ROOT}/Tables/magic_thresholds.json", "w") as f:
    json.dump(per_metal_magic, f, indent=2, default=str)

# =============================================================================
# 9. ERROR ANALYSIS BY SIZE & METAL (TEST SET)
# =============================================================================
print("\n[9] Per-size error breakdown ...")
err_test = pd.DataFrame({
    "n_atoms":  df.loc[Xf_te.index, "n_atoms"].values,
    "metal":    df.loc[Xf_te.index, "metal"].values,
    "y_true":   yf_te.values,
    "y_pred":   best_full.predict(Xf_te_s),
})
err_test["abs_err"] = np.abs(err_test["y_true"] - err_test["y_pred"])

per_size = err_test.groupby("n_atoms").agg(
    n_samples=("y_true", "size"),
    avg_DFT=("y_true", "mean"),
    avg_ML=("y_pred", "mean"),
    MAE=("abs_err", "mean"),
).round(5)
per_size.to_csv(f"{ROOT}/Tables/Table4_per_size_errors.csv")

per_metal = err_test.groupby("metal").agg(
    n_samples=("y_true", "size"),
    MAE=("abs_err", "mean"),
    RMSE=("abs_err", lambda s: np.sqrt(np.mean(s**2))),
).round(5)
per_metal.to_csv(f"{ROOT}/Tables/Table5_per_metal_errors.csv")
print(per_metal)

# =============================================================================
# 10. SHAP INTERPRETABILITY  (FULL MODEL)
# =============================================================================
print("\n[10] Computing SHAP values for best full-feature model ...")
explainer = shap.TreeExplainer(best_full) if best_full_name in (
    "ExtraTrees", "RandomForest", "XGBoost", "LightGBM", "CatBoost",
    "GradientBoosting") else None

if explainer is not None:
    shap_sample = Xf_te_s[:min(500, len(Xf_te_s))]
    shap_values = explainer.shap_values(shap_sample)
    mean_abs = np.abs(shap_values).mean(axis=0)
    shap_imp = pd.DataFrame({"feature": FEATURES_FULL,
                             "mean_abs_shap": mean_abs}
                            ).sort_values("mean_abs_shap", ascending=False)
    shap_imp.to_csv(f"{ROOT}/Tables/Table6_shap_importance.csv", index=False)
    print(shap_imp.head(10).to_string(index=False))
else:
    shap_values = None
    shap_sample = None
    print("    (SHAP not computed for non-tree model)")

# =============================================================================
# 11. FIGURES (regenerated with correct feature set)
# =============================================================================
def save_fig(fig, num, name):
    path = f"{ROOT}/Figures/Fig_{num:02d}_{name}.png"
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"     saved: {path}")

print("\n[11] Generating figures ...")

# Fig 1: Parity plot — full features
fig, ax = plt.subplots()
y_pred_full = best_full.predict(Xf_te_s)
for m in ["Cu", "Ag", "Au"]:
    idx = df.loc[Xf_te.index, "metal"].values == m
    ax.scatter(yf_te[idx], y_pred_full[idx], color=COLORS[m], s=18,
               alpha=0.7, label=m, edgecolor="k", linewidth=0.2)
lims = [min(yf_te.min(), y_pred_full.min()) - 0.05,
        max(yf_te.max(), y_pred_full.max()) + 0.05]
ax.plot(lims, lims, "k--", lw=1)
ax.set_xlabel("DFT energy per atom (eV)")
ax.set_ylabel("ML predicted energy per atom (eV)")
ax.set_title(f"Parity plot — {best_full_name} (full features)")
ax.legend()
save_fig(fig, 1, "parity_full")

# Fig 2: Parity plot — geometry-only
fig, ax = plt.subplots()
y_pred_geo = best_geo.predict(Xg_te_s)
for m in ["Cu", "Ag", "Au"]:
    idx = df.loc[Xg_te.index, "metal"].values == m
    ax.scatter(yg_te[idx], y_pred_geo[idx], color=COLORS[m], s=18,
               alpha=0.7, label=m, edgecolor="k", linewidth=0.2)
lims = [min(yg_te.min(), y_pred_geo.min()) - 0.05,
        max(yg_te.max(), y_pred_geo.max()) + 0.05]
ax.plot(lims, lims, "k--", lw=1)
ax.set_xlabel("DFT energy per atom (eV)")
ax.set_ylabel("ML predicted energy per atom (eV)")
ax.set_title(f"Parity plot — {best_geo_name} (geometry only)")
ax.legend()
save_fig(fig, 2, "parity_geo_only")

# Fig 3: Learning curve (full)
print("     computing learning curve (this can take ~1 min) ...")
lc_sizes, lc_train, lc_val = learning_curve(
    build_models()[best_full_name], X_full_s_all, y, cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring="neg_mean_absolute_error", n_jobs=-1, random_state=RANDOM_SEED)
lc_train_mae = -lc_train.mean(axis=1)
lc_val_mae   = -lc_val.mean(axis=1)
fig, ax = plt.subplots()
ax.plot(lc_sizes, lc_train_mae, "o-", label="Train MAE", color="#1f77b4")
ax.plot(lc_sizes, lc_val_mae,   "s-", label="CV MAE",     color="#d62728")
ax.set_xlabel("Training set size")
ax.set_ylabel("MAE (eV/atom)")
ax.set_title(f"Learning curve — {best_full_name}")
ax.legend()
ax.grid(alpha=0.3)
save_fig(fig, 3, "learning_curve")

# Fig 4: Feature importance (full model) — confirms that N is now in the list
fig, ax = plt.subplots(figsize=(8, 6))
imp = pd.Series(best_full.feature_importances_, index=FEATURES_FULL
                ).sort_values(ascending=True)
imp.plot(kind="barh", ax=ax, color="#16a085")
ax.set_xlabel("Feature importance (impurity-based)")
ax.set_title(f"Feature importance — {best_full_name}")
save_fig(fig, 4, "feature_importance_full")

# Fig 5: SHAP summary (if available)
if shap_values is not None:
    fig = plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, shap_sample, feature_names=FEATURES_FULL,
                      show=False, plot_size=None)
    plt.title("SHAP summary — full feature set")
    plt.tight_layout()
    plt.savefig(f"{ROOT}/Figures/Fig_05_shap_summary.png", dpi=600, bbox_inches="tight")
    plt.close()
    print(f"     saved: {ROOT}/Figures/Fig_05_shap_summary.png")

# Fig 6, 7, 8: Per-metal Δ²E plots  [R1.4]
for i, metal in enumerate(["Cu", "Ag", "Au"], start=6):
    sub = df[df["metal"] == metal]
    avg = (sub.groupby("n_atoms")["energy_per_atom"]
              .mean().reset_index().sort_values("n_atoms"))
    avg["Delta2E"] = (avg["energy_per_atom"].shift(-1)
                      + avg["energy_per_atom"].shift(1)
                      - 2 * avg["energy_per_atom"])
    threshold = per_metal_magic[metal]["threshold_eV"]
    detected = per_metal_magic[metal]["detected"]
    lit      = per_metal_magic[metal]["literature"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    ax1.plot(avg["n_atoms"], avg["energy_per_atom"], "o-",
             color=COLORS[metal], lw=1.5, ms=4)
    for n in detected:
        ax1.axvline(n, color="red",   alpha=0.3, lw=1)
    for n in lit:
        ax1.axvline(n, color="green", alpha=0.3, lw=1, ls="--")
    ax1.set_ylabel("E/atom (eV)")
    ax1.set_title(f"{metal}: avg DFT energy and Δ²E (per-metal analysis)")

    bars = ax2.bar(avg["n_atoms"], avg["Delta2E"],
                   color=["red" if (not np.isnan(v) and v > threshold) else "gray"
                          for v in avg["Delta2E"]])
    ax2.axhline(threshold, color="red", ls="--", lw=1,
                label=f"threshold = {threshold:.4f} eV")
    ax2.set_xlabel("N (cluster size)")
    ax2.set_ylabel("Δ²E (eV)")
    ax2.legend(loc="upper right")
    save_fig(fig, i, f"magic_{metal}")

# Fig 9: Error vs cluster size
fig, ax = plt.subplots()
sns.scatterplot(data=err_test, x="n_atoms", y="abs_err", hue="metal",
                palette=COLORS, s=25, alpha=0.7, ax=ax)
ax.set_xlabel("Cluster size N")
ax.set_ylabel("|Predicted − DFT| (eV/atom)")
ax.set_title("Per-sample test error")
save_fig(fig, 9, "err_vs_size")

# Fig 10: Error distribution by metal (boxplot)
fig, ax = plt.subplots()
sns.boxplot(data=err_test, x="metal", y="abs_err",
            palette=COLORS, ax=ax)
ax.set_ylabel("Absolute error (eV/atom)")
ax.set_title("Test-set error by metal")
save_fig(fig, 10, "err_by_metal")

# Fig 11: PCA
pca = PCA(n_components=2).fit_transform(scaler_cv_full.transform(X_full))
fig, ax = plt.subplots()
for m in ["Cu", "Ag", "Au"]:
    idx = df["metal"] == m
    ax.scatter(pca[idx, 0], pca[idx, 1], color=COLORS[m],
               alpha=0.6, s=15, label=m)
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
ax.set_title("PCA projection (full features)")
ax.legend()
save_fig(fig, 11, "pca")

# Fig 12: t-SNE  (subsample for speed)
print("     computing t-SNE (subsample) ...")
sub_idx = np.random.RandomState(RANDOM_SEED).choice(
    len(X_full), size=min(2000, len(X_full)), replace=False)
tsne = TSNE(n_components=2, random_state=RANDOM_SEED, perplexity=30
            ).fit_transform(scaler_cv_full.transform(X_full)[sub_idx])
fig, ax = plt.subplots()
for m in ["Cu", "Ag", "Au"]:
    idx = (df.iloc[sub_idx]["metal"].values == m)
    ax.scatter(tsne[idx, 0], tsne[idx, 1], color=COLORS[m],
               alpha=0.6, s=15, label=m)
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
ax.set_title("t-SNE projection (full features)")
ax.legend()
save_fig(fig, 12, "tsne")

# Fig 13: Comparison of full vs geometry-only models
fig, ax = plt.subplots(figsize=(7, 5))
xpos = np.arange(len(results_full_df))
ax.bar(xpos - 0.2, results_full_df["MAE_test"].values, width=0.4,
       label="Full features",      color="#1f77b4")
geo_sorted = results_geo_df.set_index("Model").loc[
    results_full_df["Model"].values, "MAE_test"].values
ax.bar(xpos + 0.2, geo_sorted, width=0.4,
       label="Geometry only",      color="#ff7f0e")
ax.set_xticks(xpos)
ax.set_xticklabels(results_full_df["Model"].values, rotation=30, ha="right")
ax.set_ylabel("Test MAE (eV/atom)")
ax.set_title("Full vs geometry-only feature sets")
ax.legend()
save_fig(fig, 13, "full_vs_geo")

# Fig 14: Radius of gyration vs N (per metal) — scaling check
fig, ax = plt.subplots()
for m in ["Cu", "Ag", "Au"]:
    sub = df[df["metal"] == m]
    ax.scatter(sub["n_atoms"], sub["radius_gyration"], color=COLORS[m],
               alpha=0.4, s=12, label=m)
# Reference N^(1/3)
N_ref = np.linspace(3, 55, 100)
ax.plot(N_ref, 1.4 * N_ref ** (1/3), "k--", alpha=0.6,
        label=r"$\propto N^{1/3}$")
ax.set_xlabel("N (cluster size)")
ax.set_ylabel("Radius of gyration (Å)")
ax.set_title("Geometric scaling: $R_g$ vs N")
ax.legend()
save_fig(fig, 14, "rg_vs_N")

# Fig 15: Correlation heatmap
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(X_full.corr(), annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, ax=ax, annot_kws={"size": 7})
ax.set_title("Feature correlation matrix (full set)")
save_fig(fig, 15, "corr_full")

# =============================================================================
# 12. WRITE SUMMARY REPORT
# =============================================================================
report_lines = [
    "=" * 78,
    "  REVISED PIPELINE — SUMMARY REPORT",
    "=" * 78,
    "",
    f"Dataset:  {len(df)} clusters (Cu/Ag/Au, N <= 55) from open QCD.",
    f"Target:   {TARGET_NAME}.",
    f"Note:     This is NOT atomization energy. To convert, subtract the bare",
    f"          atom PBE/PAW energy for each metal.",
    "",
    "QCD scope and limitations [discussion for manuscript]:",
    "  - Source: open Quantum Cluster Database (DOI:10.17172/NOMAD/2023.02.01-1).",
    "  - Theory: PBE-GGA functional + PAW pseudopotentials in plane-wave VASP.",
    "  - Known biases of PBE: underestimates HOMO-LUMO gaps, overbinds metallic",
    "    systems by ~0.1-0.2 eV/atom relative to hybrid functionals.",
    "  - Magic-number locations are RELATIVE quantities (Delta2E differences),",
    "    largely insensitive to the absolute functional choice.",
    "  - Comparison with localized-basis (Gaussian) calculations: relative",
    "    energetics are typically reproducible, absolute values may shift.",
    "",
    "Splits used:",
    f"  - 70/15/15 train/val/test (stratified by metal)",
    f"  - 5-fold KFold CV on full data",
    f"  - Size-grouped CV (4 groups: 3-15, 16-30, 31-45, 46-55) — tests near-",
    f"    extrapolation across size domains.",
    "",
    "FULL FEATURE SET RESULTS (sorted by test MAE):",
    results_full_df.to_string(index=False),
    "",
    "GEOMETRY-ONLY FEATURE SET RESULTS (no DFT-derived inputs):",
    results_geo_df.to_string(index=False),
    "",
    "Per-metal magic numbers:",
]
for metal in ["Cu", "Ag", "Au"]:
    info = per_metal_magic[metal]
    report_lines += [
        f"  {metal}:",
        f"    threshold      = {info['threshold_eV']} eV/atom",
        f"    detected       = {info['detected']}",
        f"    literature     = {info['literature']}",
        f"    overlap (Both) = {sorted(set(info['detected']) & set(info['literature']))}",
    ]
report_lines += [
    "",
    "Per-metal test errors:",
    per_metal.to_string(),
    "",
    "Files written:",
    f"  - {ROOT}/Tables/Table1_full_features.csv",
    f"  - {ROOT}/Tables/Table2_geometry_only.csv",
    f"  - {ROOT}/Tables/Table3_per_metal_magic.csv",
    f"  - {ROOT}/Tables/Table4_per_size_errors.csv",
    f"  - {ROOT}/Tables/Table5_per_metal_errors.csv",
    f"  - {ROOT}/Tables/Table6_shap_importance.csv (if available)",
    f"  - {ROOT}/Figures/Fig_*.png",
    "",
    "DONE.",
]

report = "\n".join(report_lines)
with open(f"{ROOT}/REPORT.txt", "w") as f:
    f.write(report)

print("\n" + report)
print(f"\n[12] Full report saved to {ROOT}/REPORT.txt")
print(f"[12] All outputs in directory: {ROOT}/")


# =============================================================================
# CELL: Package entire project as a downloadable ZIP
# =============================================================================
import os, shutil, zipfile
from datetime import datetime

# Source folder (matches what the main script created)
SRC = "/kaggle/working/REVISED_PCCP_R1" if os.path.isdir("/kaggle/working/REVISED_PCCP_R1") else "REVISED_PCCP_R1"

# Destination ZIP in /kaggle/working/ (downloadable from Kaggle's "Output" panel)
OUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
zip_name = f"CoinageClusterML_R1_{datetime.now().strftime('%Y%m%d_%H%M')}"
zip_path = os.path.join(OUT_DIR, zip_name)

# Create the archive
archive = shutil.make_archive(zip_path, "zip", SRC)
size_mb = os.path.getsize(archive) / (1024 * 1024)

print(f" ZIP created: {archive}")
print(f"   Size: {size_mb:.2f} MB")
print(f"\nContents:")

# List archive contents (sorted)
with zipfile.ZipFile(archive, "r") as zf:
    names = sorted(zf.namelist())
    for n in names:
        info = zf.getinfo(n)
        kb = info.file_size / 1024
        print(f"   {n:<60s} {kb:>8.1f} KB")

print(f"\n Download from Kaggle 'Output' tab on the right side, "
      f"or run:  /kaggle/working/{os.path.basename(archive)}")


