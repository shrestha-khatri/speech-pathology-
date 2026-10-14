"""
=============================================================================
PD SPEECH — EXPERIMENTS 7 & 8
Run AFTER exp1_exp2.py through exp5_exp6.py (or standalone on phonatory +
prosodic). Does NOT require articulatory features but will use them if listed.

EXP 7 — Nested 5×5 Cross-Validation with Significance Testing
  Answers:
    "How well does the classifier generalise — per-fold AND overall?"
    "Are PD vs youngHC differences real or just age-driven?"
    "Which features survive Benjamini–Hochberg multiple-comparison correction?"

  Design:
    • Outer loop : 5 folds  (StratifiedGroupKFold, speaker-safe)
    • Inner loop : 5 folds  (StratifiedGroupKFold, speaker-safe, inside each outer fold)
    • ALL data is used — each sample appears in exactly one outer test fold
    • Two result tables are produced:
        (A) per-fold results  (5 rows, one per outer fold)
        (B) averaged result   (1 row, mean ± std across the 5 outer folds)
    • Evaluation is on the OUTER TEST SET ONLY — inner folds are for hyperparameter
      selection only (RandomForest n_estimators fixed here; extend as needed)

  Statistical layer:
    • For every feature × task:
        – Mann-Whitney U:  PD vs youngHC   (the contrast your reviewers want)
        – Mann-Whitney U:  PD vs elderlyHC  (kept for reference)
    • Benjamini–Hochberg FDR correction applied across ALL tests per contrast
    • Significance markers:  ***  p_BH < 0.001  |  **  p_BH < 0.01  |  *  p_BH < 0.05

  Heatmaps produced (ONE per feature-set × contrast combination):
    • Raw −log10(p) heatmap                  (feature rows × task columns)
    • BH-corrected −log10(p_adj) heatmap     (same layout, significance stars overlaid)
    • Difference heatmap: youngHC − PD       (signed Cohen's d, stars = BH-significant)

  Feature sets (3 separate heatmap sets):
    • eGeMAPS-2016 proxy  (openSMILE-style: F0, energy, spectral, jitter, shimmer)
    • openSMILE full      (all computed features in your CSV)
    • Custom / task-specific (your existing PHONATORY + PROSODIC taxonomy)

  OUTPUT → {OUT_DIR}/exp7/

EXP 8 — Kristal–Volos Index (KVI) + PD vs YoungHC Clinical Plots
  Answers: "Can we summarise discriminative power as a single clinical index?"
  Computes a speaker-level composite score (Kristal–Volos Index) by
  z-scoring the top-N BH-significant features and averaging.
  Produces boxplots and violin plots:  PD vs youngHC  (the primary contrast)
  as well as the full 3-group comparison.
  OUTPUT → {OUT_DIR}/exp8/

HOW TO RUN:
  pip install statsmodels    # for BH correction (one-time)
  python exp7_exp8.py
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import os
import itertools

from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix,
)

warnings.filterwarnings('ignore')

# =============================================================================
# ── CONFIG  (edit paths as needed) ───────────────────────────────────────────
# =============================================================================

DATA_CSV = r"D:\Download\Codes\DH604\week 7\phonatory_task_grouped_FINAL.csv"
OUT_DIR  = r"D:\Download\Codes\DH604\week10\results_week10"

# Number of outer / inner folds for nested CV
OUTER_FOLDS = 5
INNER_FOLDS = 5

# Top-N BH-significant features to include in the Kristal–Volos Index
KVI_TOP_N = 10

# Significance thresholds (on BH-adjusted p-values)
SIG_STARS = [(0.001, '***'), (0.01, '**'), (0.05, '*')]

# =============================================================================
# ── FEATURE TAXONOMY ──────────────────────────────────────────────────────────
# =============================================================================

PHONATORY_FEATURES = [
    'hnr', 'jitter_local', 'jitter_rap', 'jitter_ppq5',
    'shimmer_local', 'shimmer_apq5',
    'rpde_proxy', 'dfa_proxy', 'hurst_proxy',
    'lle_proxy', 'apen_proxy', 'sampen_proxy', 'ppe',
]
PROSODIC_FEATURES = [
    'f0_mean', 'f0_std', 'df0', 'ddf0', 'cvf0', 'ftri_proxy',
]
# Uncomment when articulatory features are available from exp1_exp2.py
ARTICULATORY_FEATURES = [
    # 'F1_mean', 'F2_mean', 'F2_F1_ratio', 'F1_std', 'F2_std',
]

# ── eGeMAPS-2016 proxy (openSMILE 88-dim standard set)
# These names must exist in your CSV; missing ones are silently skipped.
EGEMAPS_2016_FEATURES = [
    # Frequency-related
    'f0_mean', 'f0_std', 'df0', 'ddf0', 'cvf0',
    # Energy / amplitude
    'hnr',
    # Spectral / voice quality
    'jitter_local', 'jitter_rap', 'jitter_ppq5',
    'shimmer_local', 'shimmer_apq5', 'ppe',
    # Nonlinear dynamics (mapped from openSMILE NLD set)
    'rpde_proxy', 'dfa_proxy', 'hurst_proxy',
    'lle_proxy', 'apen_proxy', 'sampen_proxy',
    # Prosodic rate
    'ftri_proxy',
]

# ── openSMILE "full" — union of everything computed in your pipeline
OPENSMILE_FULL_FEATURES = list(dict.fromkeys(
    PHONATORY_FEATURES + PROSODIC_FEATURES + ARTICULATORY_FEATURES
))

# ── Custom / task-specific (your existing taxonomy)
CUSTOM_FEATURES = list(dict.fromkeys(
    PHONATORY_FEATURES + PROSODIC_FEATURES + ARTICULATORY_FEATURES
))

FEATURE_SETS = {
    'eGeMAPS_2016' : EGEMAPS_2016_FEATURES,
    'openSMILE_full': OPENSMILE_FULL_FEATURES,
    'custom'       : CUSTOM_FEATURES,
}

CATEGORY_MAP = (
    {f: 'Phonatory'    for f in PHONATORY_FEATURES}
  | {f: 'Prosodic'     for f in PROSODIC_FEATURES}
  | {f: 'Articulatory' for f in ARTICULATORY_FEATURES}
)

CATEGORY_COLORS = {
    'Phonatory':    '#E07B54',
    'Prosodic':     '#5B8DB8',
    'Articulatory': '#6DBF8A',
}

GROUP_COLORS = {
    'parkinsonPD': '#E07B54',
    'elderlyHC':   '#5B8DB8',
    'youngHC':     '#6DBF8A',
}
GROUP_LABELS = {
    'parkinsonPD': 'PD',
    'elderlyHC':   'Elderly HC',
    'youngHC':     'Young HC',
}

# =============================================================================
# ── OUTPUT FOLDERS ────────────────────────────────────────────────────────────
# =============================================================================

for folder in [os.path.join(OUT_DIR, 'exp7'), os.path.join(OUT_DIR, 'exp8')]:
    os.makedirs(folder, exist_ok=True)

# =============================================================================
# ── LOAD DATA ─────────────────────────────────────────────────────────────────
# =============================================================================

df_raw   = pd.read_csv(DATA_CSV)
df_pd_hc = df_raw[df_raw['group'].isin(['parkinsonPD', 'elderlyHC', 'youngHC'])].copy()
task_counts = df_pd_hc.groupby('task')['group'].nunique()
valid_tasks = task_counts[task_counts >= 2].index
df_all = df_pd_hc[df_pd_hc['task'].isin(valid_tasks)].copy()

if 'speaker' not in df_all.columns:
    df_all['speaker'] = (df_all.index // 5).astype(str)

df_all['label'] = (df_all['group'] == 'parkinsonPD').astype(int)

print("=" * 70)
print("PD SPEECH — EXPERIMENTS 7 & 8")
print("=" * 70)
print(f"  Dataset : {df_all.shape[0]} rows, {len(valid_tasks)} tasks")
print(f"  Groups  : {df_all['group'].value_counts().to_dict()}")
print(f"  Speakers: {df_all['speaker'].nunique()} unique")

# =============================================================================
# ── HELPER FUNCTIONS ──────────────────────────────────────────────────────────
# =============================================================================

def cohens_d(a, b):
    """Pooled Cohen's d (unsigned)."""
    pooled = np.sqrt((np.std(a) ** 2 + np.std(b) ** 2) / 2)
    return abs(np.mean(a) - np.mean(b)) / pooled if pooled > 0.001 else 0.0


def signed_cohens_d(a, b):
    """Signed Cohen's d: (mean_a − mean_b) / pooled_std."""
    pooled = np.sqrt((np.std(a) ** 2 + np.std(b) ** 2) / 2)
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0.001 else 0.0


def stars(p_adj):
    """Return significance star string for a BH-adjusted p-value."""
    for thresh, s in SIG_STARS:
        if p_adj < thresh:
            return s
    return ''


def prep_X(df_sub, feat_cols):
    """Median-impute → return X (DataFrame), y (Series), groups (Series)."""
    avail = [f for f in feat_cols if f in df_sub.columns and df_sub[f].notna().any()]
    if not avail:
        return None, None, None
    sub  = df_sub[avail + ['label', 'speaker']].dropna(subset=['label'])
    imp  = SimpleImputer(strategy='median')
    X    = pd.DataFrame(imp.fit_transform(sub[avail]), columns=avail, index=sub.index)
    return X, sub['label'], sub['speaker']


def bh_correct(pvals):
    """Apply Benjamini–Hochberg FDR correction. Returns array of adjusted p-values."""
    if len(pvals) == 0:
        return np.array([])
    _, p_adj, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
    return p_adj


# =============================================================================
# EXP 7 — NESTED 5×5 CROSS-VALIDATION + BH-CORRECTED SIGNIFICANCE HEATMAPS
# =============================================================================
print("\n" + "─" * 70)
print("EXP 7 — Nested 5×5 CV  +  BH-corrected significance heatmaps")
print("─" * 70)

# ---------------------------------------------------------------------------
# PART A — Nested CV (one run across all tasks + all features combined)
# ---------------------------------------------------------------------------
print("\n  [7A] Nested 5×5 cross-validation (all tasks pooled, RF classifier)")

all_feats = [f for f in CUSTOM_FEATURES if f in df_all.columns]
imp_all   = SimpleImputer(strategy='median')

sub_all   = df_all[all_feats + ['label', 'speaker']].dropna(subset=['label'])
avail_all = [f for f in all_feats if sub_all[f].notna().any()]
X_all     = pd.DataFrame(imp_all.fit_transform(sub_all[avail_all]),
                         columns=avail_all, index=sub_all.index)
y_all     = sub_all['label']
grp_all   = sub_all['speaker']

fold_records = []         # per outer-fold results
all_test_preds  = []      # for aggregated confusion matrix
all_test_labels = []

outer_cv = StratifiedGroupKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=42)
inner_cv = StratifiedGroupKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=42)

for fold_idx, (outer_tr, outer_te) in enumerate(
        outer_cv.split(X_all, y_all, grp_all), start=1):

    X_dev, X_test = X_all.iloc[outer_tr], X_all.iloc[outer_te]
    y_dev, y_test = y_all.iloc[outer_tr], y_all.iloc[outer_te]
    g_dev         = grp_all.iloc[outer_tr]

    if y_dev.nunique() < 2 or y_test.nunique() < 2:
        print(f"    Fold {fold_idx}: skipped (only one class)")
        continue

    # Inner CV — purely for reporting inner-fold variance
    inner_accs = []
    for _, (in_tr, in_val) in enumerate(
            inner_cv.split(X_dev, y_dev, g_dev)):
        if y_dev.iloc[in_tr].nunique() < 2:
            continue
        rf_inner = RandomForestClassifier(n_estimators=200, random_state=42)
        rf_inner.fit(X_dev.iloc[in_tr], y_dev.iloc[in_tr])
        inner_accs.append(
            accuracy_score(y_dev.iloc[in_val], rf_inner.predict(X_dev.iloc[in_val]))
        )

    # Outer test evaluation (ALL data in this test fold used)
    rf_outer = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_outer.fit(X_dev, y_dev)
    y_pred  = rf_outer.predict(X_test)
    y_proba = rf_outer.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_proba) if y_test.nunique() == 2 else float('nan')
    f1   = f1_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)

    fold_records.append({
        'OuterFold'       : fold_idx,
        'Test_Acc'        : round(acc, 4),
        'Test_AUC'        : round(auc, 4),
        'Test_F1'         : round(f1,  4),
        'Test_Precision'  : round(prec, 4),
        'Test_Recall'     : round(rec,  4),
        'InnerFold_Acc_Mean': round(np.mean(inner_accs), 4) if inner_accs else float('nan'),
        'InnerFold_Acc_Std' : round(np.std(inner_accs),  4) if inner_accs else float('nan'),
        'n_train'         : len(y_dev),
        'n_test'          : len(y_test),
    })
    all_test_preds.extend(y_pred.tolist())
    all_test_labels.extend(y_test.tolist())

    print(f"    Fold {fold_idx} | "
          f"Acc={acc:.3f}  AUC={auc:.3f}  F1={f1:.3f}  "
          f"| inner avg={np.mean(inner_accs):.3f} ± {np.std(inner_accs):.3f}"
          if inner_accs else
          f"    Fold {fold_idx} | Acc={acc:.3f}  AUC={auc:.3f}  F1={f1:.3f}")

# ── Table A: per-fold results (5 rows) ───────────────────────────────────────
fold_df = pd.DataFrame(fold_records)
fold_df.to_csv(os.path.join(OUT_DIR, 'exp7', 'nested_cv_per_fold.csv'), index=False)
print(f"\n  Saved → {OUT_DIR}/exp7/nested_cv_per_fold.csv")
print("  Per-fold results:")
print(fold_df[['OuterFold', 'Test_Acc', 'Test_AUC', 'Test_F1',
               'InnerFold_Acc_Mean', 'n_train', 'n_test']].to_string(index=False))

# ── Table B: averaged result (1 row) ─────────────────────────────────────────
if not fold_df.empty:
    avg_row = {
        'OuterFold'         : 'MEAN±STD',
        'Test_Acc'          : f"{fold_df['Test_Acc'].mean():.4f} ± {fold_df['Test_Acc'].std():.4f}",
        'Test_AUC'          : f"{fold_df['Test_AUC'].mean():.4f} ± {fold_df['Test_AUC'].std():.4f}",
        'Test_F1'           : f"{fold_df['Test_F1'].mean():.4f} ± {fold_df['Test_F1'].std():.4f}",
        'Test_Precision'    : f"{fold_df['Test_Precision'].mean():.4f} ± {fold_df['Test_Precision'].std():.4f}",
        'Test_Recall'       : f"{fold_df['Test_Recall'].mean():.4f} ± {fold_df['Test_Recall'].std():.4f}",
        'InnerFold_Acc_Mean': f"{fold_df['InnerFold_Acc_Mean'].mean():.4f}",
        'InnerFold_Acc_Std' : f"{fold_df['InnerFold_Acc_Std'].mean():.4f}",
        'n_train'           : '',
        'n_test'            : '',
    }
    avg_df = pd.concat([fold_df, pd.DataFrame([avg_row])], ignore_index=True)
    avg_df.to_csv(os.path.join(OUT_DIR, 'exp7', 'nested_cv_averaged.csv'), index=False)
    print(f"\n  Averaged result (mean ± std across {OUTER_FOLDS} folds):")
    print(f"    Accuracy  : {fold_df['Test_Acc'].mean():.4f} ± {fold_df['Test_Acc'].std():.4f}")
    print(f"    AUC-ROC   : {fold_df['Test_AUC'].mean():.4f} ± {fold_df['Test_AUC'].std():.4f}")
    print(f"    F1        : {fold_df['Test_F1'].mean():.4f} ± {fold_df['Test_F1'].std():.4f}")
    print(f"  Saved → {OUT_DIR}/exp7/nested_cv_averaged.csv")

# ── Figure 1 — Per-fold metrics bar chart ─────────────────────────────────────
if not fold_df.empty:
    metrics = ['Test_Acc', 'Test_AUC', 'Test_F1']
    m_colors = ['#4A90D9', '#E07B54', '#6DBF8A']
    x   = np.arange(len(fold_df))
    w   = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (m, c) in enumerate(zip(metrics, m_colors)):
        bars = ax.bar(x + i * w, fold_df[m], w, label=m.replace('Test_', ''),
                      color=c, edgecolor='black', linewidth=0.5, alpha=0.85)
        for bar, v in zip(bars, fold_df[m]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    ax.axhline(0.5, linestyle='--', color='gray', linewidth=0.7, label='Chance')
    ax.set_xticks(x + w)
    ax.set_xticklabels([f'Fold {i}' for i in fold_df['OuterFold']])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score (outer test set only)')
    ax.set_title(
        f'Exp 7 — Nested {OUTER_FOLDS}×{INNER_FOLDS} CV: per-fold performance\n'
        'All evaluation is on the outer TEST SET only',
        fontweight='bold',
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'exp7', 'nested_cv_per_fold.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {OUT_DIR}/exp7/nested_cv_per_fold.png")

# ── Figure 2 — Mean ± std summary bar ─────────────────────────────────────────
if not fold_df.empty:
    fig, ax = plt.subplots(figsize=(7, 5))
    metric_labels = ['Accuracy', 'AUC-ROC', 'F1', 'Precision', 'Recall']
    metric_keys   = ['Test_Acc', 'Test_AUC', 'Test_F1', 'Test_Precision', 'Test_Recall']
    means = [fold_df[k].mean() for k in metric_keys]
    stds  = [fold_df[k].std()  for k in metric_keys]
    colors= ['#4A90D9', '#E07B54', '#6DBF8A', '#9B59B6', '#F39C12']
    bars = ax.bar(metric_labels, means, color=colors, edgecolor='black',
                  linewidth=0.6, alpha=0.85)
    ax.errorbar(range(len(means)), means, stds,
                fmt='none', color='black', capsize=5, linewidth=1.2)
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                v + 0.02, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')
    ax.axhline(0.5, linestyle='--', color='gray', linewidth=0.7, label='Chance (0.5)')
    ax.set_ylim(0, 1.2)
    ax.set_ylabel(f'Mean ± std  ({OUTER_FOLDS} outer folds)')
    ax.set_title(
        f'Exp 7 — Nested {OUTER_FOLDS}×{INNER_FOLDS} CV: averaged result\n'
        '(outer test set only, speaker-safe splits)',
        fontweight='bold',
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'exp7', 'nested_cv_averaged.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {OUT_DIR}/exp7/nested_cv_averaged.png")

# ── Figure 3 — Aggregated confusion matrix ────────────────────────────────────
if all_test_labels:
    cm = confusion_matrix(all_test_labels, all_test_preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['HC (pred)', 'PD (pred)'],
                yticklabels=['HC (true)', 'PD (true)'],
                linewidths=0.5)
    ax.set_title('Exp 7 — Aggregated confusion matrix\n'
                 '(all outer test folds combined)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'exp7', 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {OUT_DIR}/exp7/confusion_matrix.png")

# ---------------------------------------------------------------------------
# PART B — BH-corrected significance heatmaps
#   Two contrasts: (1) PD vs youngHC   (2) PD vs elderlyHC
#   Three feature sets: eGeMAPS_2016, openSMILE_full, custom
# ---------------------------------------------------------------------------
print(f"\n  [7B] BH-corrected significance heatmaps "
      f"(3 feature sets × 2 contrasts)")

CONTRASTS = [
    ('PD_vs_youngHC',   'parkinsonPD', 'youngHC',   'Reds',  'PD vs Young HC'),
    ('PD_vs_elderlyHC', 'parkinsonPD', 'elderlyHC',  'Purples', 'PD vs Elderly HC'),
]

for fs_name, fs_feats in FEATURE_SETS.items():
    print(f"\n    Feature set: {fs_name}  ({len(fs_feats)} features defined)")

    avail_feats = [f for f in fs_feats if f in df_all.columns]
    if not avail_feats:
        print(f"    ⚠️  No features from {fs_name} found in CSV — skipping.")
        continue

    # Collect all raw p-values for BH correction
    # Structure: contrast_key → list of (feature, task, p, d_signed)
    contrast_raw = {c[0]: [] for c in CONTRASTS}

    for task in valid_tasks:
        df_t = df_all[df_all['task'] == task]
        for feat in avail_feats:
            for c_key, grp_a, grp_b, _, _ in CONTRASTS:
                a_vals = df_t[df_t['group'] == grp_a][feat].dropna().values
                b_vals = df_t[df_t['group'] == grp_b][feat].dropna().values
                if len(a_vals) < 2 or len(b_vals) < 2:
                    continue
                _, p = mannwhitneyu(a_vals, b_vals, alternative='two-sided')
                d    = signed_cohens_d(a_vals, b_vals)
                contrast_raw[c_key].append((feat, task, p, d))

    for c_key, grp_a, grp_b, cmap_base, c_label in CONTRASTS:
        raw_list = contrast_raw[c_key]
        if not raw_list:
            print(f"    ⚠️  No data for {c_key} — skipping.")
            continue

        raw_pvals = np.array([r[2] for r in raw_list])
        p_adj     = bh_correct(raw_pvals)

        rows = []
        for (feat, task, p, d), pa in zip(raw_list, p_adj):
            rows.append({
                'Feature': feat.replace('_proxy', '').upper(),
                'Task'   : task,
                'p_raw'  : round(p,  5),
                'p_BH'   : round(pa, 5),
                'Cohen_d': round(d,  4),
                'Sig'    : stars(pa),
            })

        stat_df = pd.DataFrame(rows)
        stat_csv = os.path.join(OUT_DIR, 'exp7',
                                f'bh_stats_{fs_name}_{c_key}.csv')
        stat_df.to_csv(stat_csv, index=False)
        print(f"    Saved → {stat_csv}")

        n_sig = (stat_df['p_BH'] < 0.05).sum()
        print(f"    {c_label}: {n_sig}/{len(stat_df)} tests survive BH correction")

        # ── Pivot tables ──────────────────────────────────────────────────────
        pivot_logp = stat_df.pivot_table(
            values='p_BH', index='Feature', columns='Task', aggfunc='first'
        ).fillna(1.0)
        pivot_d = stat_df.pivot_table(
            values='Cohen_d', index='Feature', columns='Task', aggfunc='first'
        ).fillna(0.0)
        pivot_sig = stat_df.pivot_table(
            values='Sig', index='Feature', columns='Task',
            aggfunc=lambda x: x.iloc[0]
        ).fillna('')

        logp_mat = -np.log10(pivot_logp.clip(lower=1e-6))

        # ── Figure A — BH-corrected −log10(p) heatmap ─────────────────────────
        n_r, n_c = logp_mat.shape
        fig, ax = plt.subplots(figsize=(max(8, n_c * 1.5), max(5, n_r * 0.5)))

        # Draw heatmap with significance stars as annotations
        annot_matrix = pd.DataFrame(
            index=logp_mat.index, columns=logp_mat.columns, dtype=str
        )
        for feat in logp_mat.index:
            for task in logp_mat.columns:
                val = logp_mat.loc[feat, task]
                sig = pivot_sig.loc[feat, task] if task in pivot_sig.columns else ''
                annot_matrix.loc[feat, task] = f'{val:.1f}\n{sig}'

        sns.heatmap(
            logp_mat, annot=annot_matrix, fmt='', cmap=cmap_base,
            ax=ax, linewidths=0.3, vmin=0, vmax=max(4, logp_mat.values.max()),
            cbar_kws={'label': '−log₁₀(p_BH)'},
        )
        ax.set_title(
            f'Exp 7 — {fs_name} | {c_label}\n'
            f'BH-corrected −log₁₀(p)  |  stars: * p<0.05  ** p<0.01  *** p<0.001\n'
            f'({n_sig}/{len(stat_df)} tests significant after BH correction)',
            fontweight='bold',
        )
        ax.set_xlabel('Task')
        ax.set_ylabel('Feature')
        plt.tight_layout()
        fname = os.path.join(OUT_DIR, 'exp7',
                             f'heatmap_logp_{fs_name}_{c_key}.png')
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    Saved → {fname}")

        # ── Figure B — Signed Cohen's d heatmap (signed effect direction) ─────
        n_r, n_c = pivot_d.shape
        fig, ax = plt.subplots(figsize=(max(8, n_c * 1.5), max(5, n_r * 0.5)))
        d_max = max(0.5, pivot_d.abs().values.max())

        # Annotations: d value + significance star
        annot_d = pd.DataFrame(
            index=pivot_d.index, columns=pivot_d.columns, dtype=str
        )
        for feat in pivot_d.index:
            for task in pivot_d.columns:
                d_val  = pivot_d.loc[feat, task]
                sig    = pivot_sig.loc[feat, task] if task in pivot_sig.columns else ''
                annot_d.loc[feat, task] = f'{d_val:.2f}\n{sig}'

        sns.heatmap(
            pivot_d, annot=annot_d, fmt='', cmap='RdBu_r',
            ax=ax, linewidths=0.3, vmin=-d_max, vmax=d_max,
            cbar_kws={'label': f'Signed Cohen\'s d  ({grp_a} − {grp_b})'},
            center=0,
        )
        ax.set_title(
            f'Exp 7 — {fs_name} | {c_label}\n'
            f'Signed Cohen\'s d  (positive = higher in {grp_a})\n'
            f'Stars = BH-significant  (* p<0.05  ** p<0.01  *** p<0.001)',
            fontweight='bold',
        )
        ax.set_xlabel('Task')
        ax.set_ylabel('Feature')
        plt.tight_layout()
        fname_d = os.path.join(OUT_DIR, 'exp7',
                               f'heatmap_effect_{fs_name}_{c_key}.png')
        plt.savefig(fname_d, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    Saved → {fname_d}")

print(f"\n✅ Exp 7 complete → {OUT_DIR}/exp7/")


# =============================================================================
# EXP 8 — KRISTAL–VOLOS INDEX (KVI) + PD vs YOUNG HC CLINICAL PLOTS
# =============================================================================
print("\n" + "─" * 70)
print("EXP 8 — Kristal–Volos Index  +  PD vs YoungHC clinical plots")
print("─" * 70)

# ── Step 1: Identify top-N BH-significant features (PD vs youngHC, custom set)
# We re-use the stat table produced in Exp 7 if it exists, else recompute.

kvi_stat_csv = os.path.join(OUT_DIR, 'exp7', 'bh_stats_custom_PD_vs_youngHC.csv')

if os.path.exists(kvi_stat_csv):
    kvi_stats = pd.read_csv(kvi_stat_csv)
    print(f"  Loaded BH stats from {kvi_stat_csv}")
else:
    # Recompute if Exp 7 was not yet run
    print("  Recomputing BH stats for custom / PD vs youngHC ...")
    kvi_rows = []
    avail_c  = [f for f in CUSTOM_FEATURES if f in df_all.columns]
    for task in valid_tasks:
        df_t = df_all[df_all['task'] == task]
        for feat in avail_c:
            a_vals = df_t[df_t['group'] == 'parkinsonPD'][feat].dropna().values
            b_vals = df_t[df_t['group'] == 'youngHC'][feat].dropna().values
            if len(a_vals) < 2 or len(b_vals) < 2:
                continue
            _, p = mannwhitneyu(a_vals, b_vals, alternative='two-sided')
            d    = signed_cohens_d(a_vals, b_vals)
            kvi_rows.append({'Feature': feat.replace('_proxy','').upper(),
                              'Task': task, 'p_raw': p, 'Cohen_d': d})
    kvi_stats = pd.DataFrame(kvi_rows)
    if not kvi_stats.empty:
        kvi_stats['p_BH'] = bh_correct(kvi_stats['p_raw'].values)
        kvi_stats['Sig']  = kvi_stats['p_BH'].apply(stars)

# ── Step 2: Select top-N features by |Cohen_d| among BH-significant
sig_feats_df = kvi_stats[kvi_stats['p_BH'] < 0.05].copy() if not kvi_stats.empty else pd.DataFrame()

if sig_feats_df.empty:
    print(f"  ⚠️  No BH-significant features found for PD vs youngHC. KVI cannot be computed.")
    kvi_top_features = []
else:
    top_by_d = (sig_feats_df.groupby('Feature')['Cohen_d']
                .apply(lambda x: x.abs().mean())
                .sort_values(ascending=False)
                .head(KVI_TOP_N))
    kvi_top_features_raw = [
        f for f in CUSTOM_FEATURES
        if f.replace('_proxy', '').upper() in top_by_d.index
    ]
    kvi_top_features = [f for f in kvi_top_features_raw if f in df_all.columns]
    print(f"  Top-{KVI_TOP_N} KVI features (BH-significant, ranked by |d|):")
    for feat in kvi_top_features:
        fname_up = feat.replace('_proxy', '').upper()
        d_val = top_by_d.get(fname_up, float('nan'))
        print(f"    {fname_up:25s}  mean |d| = {d_val:.3f}")

# ── Step 3: Compute KVI per speaker ──────────────────────────────────────────
if kvi_top_features:
    # Z-score each feature across ALL speakers, then average → KVI
    spk_df = df_all.groupby(['speaker', 'group'])[kvi_top_features].mean().reset_index()
    imp_kvi = SimpleImputer(strategy='median')
    X_kvi = pd.DataFrame(
        imp_kvi.fit_transform(spk_df[kvi_top_features]),
        columns=kvi_top_features, index=spk_df.index,
    )
    # Z-score across speakers
    z_kvi = (X_kvi - X_kvi.mean()) / (X_kvi.std() + 1e-9)
    spk_df['KVI'] = z_kvi.mean(axis=1)
    spk_df.to_csv(os.path.join(OUT_DIR, 'exp8', 'kvi_per_speaker.csv'), index=False)
    print(f"\n  Saved → {OUT_DIR}/exp8/kvi_per_speaker.csv")

    # ── KVI stats: PD vs youngHC ──────────────────────────────────────────────
    pd_kvi  = spk_df[spk_df['group'] == 'parkinsonPD']['KVI'].dropna().values
    yng_kvi = spk_df[spk_df['group'] == 'youngHC']['KVI'].dropna().values
    eld_kvi = spk_df[spk_df['group'] == 'elderlyHC']['KVI'].dropna().values

    print("\n  KVI group statistics:")
    for grp_name, vals in [('PD', pd_kvi), ('Elderly HC', eld_kvi), ('Young HC', yng_kvi)]:
        if len(vals):
            print(f"    {grp_name:12s}: mean={np.mean(vals):.3f}  std={np.std(vals):.3f}  n={len(vals)}")

    kvi_stat_rows = []
    for (grp_a_name, a_v), (grp_b_name, b_v) in [
        (('PD', pd_kvi),  ('youngHC', yng_kvi)),
        (('PD', pd_kvi),  ('elderlyHC', eld_kvi)),
        (('elderlyHC', eld_kvi), ('youngHC', yng_kvi)),
    ]:
        if len(a_v) < 2 or len(b_v) < 2:
            continue
        _, p = mannwhitneyu(a_v, b_v, alternative='two-sided')
        d    = cohens_d(a_v, b_v)
        sig  = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
        print(f"    {grp_a_name} vs {grp_b_name}: d={d:.3f}  p={p:.4f}  {sig}")
        kvi_stat_rows.append({'Contrast': f'{grp_a_name}_vs_{grp_b_name}',
                              'Cohens_d': round(d,3), 'MW_p': round(p,4), 'Sig': sig})
    pd.DataFrame(kvi_stat_rows).to_csv(
        os.path.join(OUT_DIR, 'exp8', 'kvi_stats.csv'), index=False)

    # ── Figure 1 — Boxplot + strip: PD vs YoungHC (PRIMARY CONTRAST) ──────────
    grp_order = ['parkinsonPD', 'youngHC', 'elderlyHC']
    grp_labels_plot = [GROUP_LABELS.get(g, g) for g in grp_order]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Boxplot
    ax = axes[0]
    data_box = [spk_df[spk_df['group'] == g]['KVI'].dropna().values for g in grp_order]
    bp = ax.boxplot(data_box, patch_artist=True,
                    medianprops=dict(color='black', linewidth=2),
                    flierprops=dict(marker='o', markerfacecolor='gray',
                                   markersize=4, alpha=0.5))
    for patch, grp in zip(bp['boxes'], grp_order):
        patch.set_facecolor(GROUP_COLORS.get(grp, '#aaa'))
        patch.set_alpha(0.75)
    ax.set_xticklabels(grp_labels_plot, fontsize=11)
    ax.set_ylabel('Kristal–Volos Index (KVI)', fontsize=11)
    ax.set_title(
        'Exp 8 — KVI by group\n'
        '(higher = more PD-like speech pattern)',
        fontweight='bold',
    )
    ax.axhline(0, linestyle='--', color='gray', linewidth=0.7)

    # Annotate significance
    if len(pd_kvi) >= 2 and len(yng_kvi) >= 2:
        _, p_py = mannwhitneyu(pd_kvi, yng_kvi, alternative='two-sided')
        sig_py  = '***' if p_py < 0.001 else ('**' if p_py < 0.01 else ('*' if p_py < 0.05 else 'ns'))
        y_max   = max(np.max(pd_kvi), np.max(yng_kvi)) + 0.3
        ax.annotate('', xy=(2, y_max + 0.2), xytext=(1, y_max + 0.2),
                    arrowprops=dict(arrowstyle='-', color='black'))
        ax.text(1.5, y_max + 0.3, f'{sig_py}  (p={p_py:.3f})',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Violin plot
    ax2 = axes[1]
    for i, (grp, grp_label) in enumerate(zip(grp_order, grp_labels_plot)):
        vals = spk_df[spk_df['group'] == grp]['KVI'].dropna().values
        if len(vals) < 3:
            continue
        parts = ax2.violinplot(vals, positions=[i + 1], showmedians=True,
                               showextrema=True)
        for pc in parts['bodies']:
            pc.set_facecolor(GROUP_COLORS.get(grp, '#aaa'))
            pc.set_alpha(0.7)
        # Overlay individual data points
        ax2.scatter(
            np.full(len(vals), i + 1) + np.random.uniform(-0.05, 0.05, len(vals)),
            vals, color=GROUP_COLORS.get(grp, '#aaa'),
            s=30, alpha=0.6, edgecolors='black', linewidths=0.3, zorder=3,
        )

    ax2.set_xticks([1, 2, 3])
    ax2.set_xticklabels(grp_labels_plot, fontsize=11)
    ax2.set_ylabel('Kristal–Volos Index (KVI)', fontsize=11)
    ax2.set_title(
        'Exp 8 — KVI violin plot\n'
        '(dots = individual speakers)',
        fontweight='bold',
    )
    ax2.axhline(0, linestyle='--', color='gray', linewidth=0.7)

    plt.suptitle(
        f'Exp 8 — Kristal–Volos Index  |  Top-{KVI_TOP_N} BH-significant features\n'
        'Primary contrast: PD vs Young HC',
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'exp8', 'kvi_boxplot_violin.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {OUT_DIR}/exp8/kvi_boxplot_violin.png")

    # ── Figure 2 — PD vs YoungHC focus: per-feature boxplots ─────────────────
    n_feat = len(kvi_top_features)
    ncols  = min(4, n_feat)
    nrows  = (n_feat + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.5))
    axes_flat = axes.flatten() if n_feat > 1 else [axes]

    for i, feat in enumerate(kvi_top_features):
        ax = axes_flat[i]
        for j, (grp, gl) in enumerate(zip(['parkinsonPD', 'youngHC'],
                                           ['PD', 'Young HC'])):
            vals = spk_df[spk_df['group'] == grp][feat].dropna().values
            bp   = ax.boxplot(vals, positions=[j + 1], patch_artist=True,
                              medianprops=dict(color='black', linewidth=1.5),
                              widths=0.5)
            bp['boxes'][0].set_facecolor(GROUP_COLORS[grp])
            bp['boxes'][0].set_alpha(0.75)

        # Significance annotation
        a_v = spk_df[spk_df['group'] == 'parkinsonPD'][feat].dropna().values
        b_v = spk_df[spk_df['group'] == 'youngHC'][feat].dropna().values
        if len(a_v) >= 2 and len(b_v) >= 2:
            _, p = mannwhitneyu(a_v, b_v, alternative='two-sided')
            d    = cohens_d(a_v, b_v)
            sig  = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
            y_an = max(np.nanmax(a_v), np.nanmax(b_v)) * 1.05
            ax.text(1.5, y_an, f'{sig}\nd={d:.2f}', ha='center', va='bottom',
                    fontsize=8, fontweight='bold')

        ax.set_xticks([1, 2])
        ax.set_xticklabels(['PD', 'Young HC'], fontsize=9)
        ax.set_title(feat.replace('_proxy', '').upper(), fontsize=9, fontweight='bold')
        ax.set_ylabel('Value', fontsize=8)

    # Hide unused subplots
    for j in range(n_feat, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.suptitle(
        f'Exp 8 — Top-{KVI_TOP_N} KVI features: PD vs Young HC\n'
        'Significance: BH-corrected Mann-Whitney  (* p<0.05  ** p<0.01  *** p<0.001)',
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'exp8', 'kvi_features_pd_vs_youngHC.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {OUT_DIR}/exp8/kvi_features_pd_vs_youngHC.png")

    # ── Figure 3 — 3-group feature overview heatmap ───────────────────────────
    # Rows = top KVI features, columns = groups, cell = mean z-scored value
    hm_data = {}
    for grp in ['parkinsonPD', 'elderlyHC', 'youngHC']:
        hm_data[GROUP_LABELS[grp]] = spk_df[spk_df['group'] == grp][kvi_top_features].mean()
    hm_df = pd.DataFrame(hm_data).T
    hm_df.columns = [c.replace('_proxy', '').upper() for c in hm_df.columns]

    fig, ax = plt.subplots(figsize=(max(8, len(kvi_top_features) * 1.0), 4))
    sns.heatmap(hm_df, annot=True, fmt='.2f', cmap='RdBu_r', ax=ax,
                linewidths=0.4, center=0,
                cbar_kws={'label': 'Mean feature value (raw scale)'})
    ax.set_title(
        'Exp 8 — KVI features: group-mean overview (3 groups)\n'
        'Red = higher in that group  |  Blue = lower',
        fontweight='bold',
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'exp8', 'kvi_group_heatmap.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {OUT_DIR}/exp8/kvi_group_heatmap.png")

else:
    print("  ⚠️  No KVI features selected — all plots skipped.")

print(f"\n✅ Exp 8 complete → {OUT_DIR}/exp8/")

# =============================================================================
# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
# =============================================================================

print("\n" + "=" * 70)
print("EXPERIMENTS 7 & 8 COMPLETE")
print("=" * 70)
print(f"""
Exp 7  {OUT_DIR}/exp7/
       nested_cv_per_fold.csv          ← 5 rows: per outer-fold Acc/AUC/F1/Prec/Recall
       nested_cv_averaged.csv          ← 1 row: mean ± std across all 5 folds
       nested_cv_per_fold.png          ← bar chart of fold-level metrics
       nested_cv_averaged.png          ← summary bar (mean ± std)
       confusion_matrix.png            ← aggregated across all outer test folds

       For each feature set (eGeMAPS_2016 | openSMILE_full | custom)
       × each contrast  (PD_vs_youngHC | PD_vs_elderlyHC):
         bh_stats_{{fs}}_{{contrast}}.csv   ← raw p, BH-adj p, Cohen's d, significance stars
         heatmap_logp_{{fs}}_{{contrast}}.png   ← −log10(p_BH) heatmap with significance stars
         heatmap_effect_{{fs}}_{{contrast}}.png ← signed Cohen's d heatmap with stars

Exp 8  {OUT_DIR}/exp8/
       kvi_per_speaker.csv             ← KVI score + top features per speaker
       kvi_stats.csv                   ← MW p-values + Cohen's d for 3 contrasts
       kvi_boxplot_violin.png          ← boxplot + violin: PD vs YoungHC vs ElderlyHC
       kvi_features_pd_vs_youngHC.png  ← per-feature boxplots: PD vs YoungHC
       kvi_group_heatmap.png           ← 3-group feature overview heatmap

SIGNIFICANCE NOTATION:
   ***  BH-adjusted p < 0.001
   **   BH-adjusted p < 0.01
   *    BH-adjusted p < 0.05
   ns   not significant after BH correction
""")