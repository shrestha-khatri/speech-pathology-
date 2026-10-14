"""
=============================================================================
PD SPEECH — EXPERIMENTS 3 & 4
Run AFTER your original pipeline AND after pd_exp1_exp2.py if you want
articulatory features included. Can also run standalone on phonatory + prosodic.

EXP 3 — Age confound analysis (3-group: PD / elderlyHC / youngHC)
  Answers: "Is this feature actually disease-specific, or is it just an ageing effect?"
  For every feature × task, runs three Mann-Whitney comparisons and classifies
  each result into one of four patterns:
    A) disease-specific  (PD ≠ elderlyHC,  elderlyHC ≈ youngHC)
    B) ageing + disease  (PD ≠ elderlyHC,  elderlyHC ≠ youngHC)
    C) ageing only       (PD ≈ elderlyHC,  elderlyHC ≠ youngHC)
    D) no effect         (neither comparison significant)
  OUTPUT → {OUT_DIR}\\exp3\\

EXP 4 — Task-type sensitivity
  Answers: "Which task type should you use to screen PD, and with which features?"
  Trains RF + SVM separately on each task × category combination and reports accuracy.
  OUTPUT → {OUT_DIR}\\exp4\\

HOW TO RUN:
  python pd_exp3_exp4.py
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

from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')

# =============================================================================
# ── CONFIG ───────────────────────────────────────────────────────────────────
# =============================================================================

DATA_CSV = r"D:\Download\Codes\DH604\week 7\phonatory_task_grouped_FINAL.csv"

# =============================================================================
# ── FEATURE TAXONOMY ─────────────────────────────────────────────────────────
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
# If you ran pd_exp1_exp2.py and Exp 2 extracted articulatory features,
# uncomment the list below. Leave empty if Exp 2 was skipped.
# These match the features computed from your nested audio folder structure:
#   F1_mean, F2_mean   — mean formant values across vowel tasks per speaker
#   F2_F1_ratio        — F2/F1 ratio; lower in PD = more centralised vowels
#   F1_std, F2_std     — formant variability; lower in PD = less articulatory movement
ARTICULATORY_FEATURES = [
    # 'F1_mean', 'F2_mean', 'F2_F1_ratio', 'F1_std', 'F2_std',
]

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

# =============================================================================
# ── OUTPUT FOLDERS ────────────────────────────────────────────────────────────
# =============================================================================

OUT_DIR = r"D:\Download\Codes\DH604\week 9\results_week9"
for folder in [os.path.join(OUT_DIR, 'exp3'), os.path.join(OUT_DIR, 'exp4')]:
    os.makedirs(folder, exist_ok=True)

# =============================================================================
# ── LOAD DATA ─────────────────────────────────────────────────────────────────
# =============================================================================

df_raw    = pd.read_csv(DATA_CSV)
df_pd_hc  = df_raw[df_raw['group'].isin(['parkinsonPD', 'elderlyHC', 'youngHC'])].copy()
task_counts = df_pd_hc.groupby('task')['group'].nunique()
valid_tasks = task_counts[task_counts >= 2].index
df_all    = df_pd_hc[df_pd_hc['task'].isin(valid_tasks)].copy()

if 'speaker' not in df_all.columns:
    df_all['speaker'] = (df_all.index // 5).astype(str)

df_all['label'] = (df_all['group'] == 'parkinsonPD').astype(int)

print("=" * 70)
print("PD SPEECH — EXPERIMENTS 3 & 4")
print("=" * 70)
print(f"  Dataset : {df_all.shape[0]} rows, {len(valid_tasks)} tasks")
print(f"  Groups  : {df_all['group'].value_counts().to_dict()}")

# =============================================================================
# ── HELPER FUNCTIONS ──────────────────────────────────────────────────────────
# =============================================================================

def cohens_d(a, b):
    pooled = np.sqrt((np.std(a) ** 2 + np.std(b) ** 2) / 2)
    return abs(np.mean(a) - np.mean(b)) / pooled if pooled > 0.001 else 0


def prep_X(df_task, feat_cols):
    """Median-impute; return X (DataFrame), y (Series), groups (Series)."""
    sub  = df_task[feat_cols + ['label', 'speaker']].dropna(subset=['label'])
    cols = [c for c in feat_cols if sub[c].notna().any()]
    imp  = SimpleImputer(strategy='median')
    X    = pd.DataFrame(imp.fit_transform(sub[cols]), columns=cols, index=sub.index)
    return X, sub['label'], sub['speaker']


def run_cv(X, y, groups, models, n_splits_max=5):
    """StratifiedGroupKFold CV. Returns {model_name: [fold_accuracies]}."""
    n = min(n_splits_max, int(y.value_counts().min()))
    if n < 2:
        return {}
    sgkf = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=42)
    out  = {name: [] for name, _ in models}
    for tr, te in sgkf.split(X, y, groups):
        if len(np.unique(y.iloc[tr])) < 2:
            continue
        for name, mdl in models:
            mdl.fit(X.iloc[tr], y.iloc[tr])
            out[name].append(accuracy_score(y.iloc[te], mdl.predict(X.iloc[te])))
    return out


# =============================================================================
# EXP 3 — AGE CONFOUND ANALYSIS
# =============================================================================
# WHAT IT DOES:
#   For every feature and every task, runs three Mann-Whitney U tests:
#     (1) PD vs elderlyHC      — is there a disease effect?
#     (2) PD vs youngHC        — is there a combined disease + ageing effect?
#     (3) elderlyHC vs youngHC — is there a pure ageing effect?
#
#   Based on which of (1) and (3) are significant, each feature×task pair is
#   classified into one of four patterns:
#     A — PD ≠ elderlyHC, elderlyHC ≈ youngHC  →  disease-specific marker
#     B — PD ≠ elderlyHC, elderlyHC ≠ youngHC  →  amplified by both ageing & disease
#     C — PD ≈ elderlyHC, elderlyHC ≠ youngHC  →  pure ageing effect (not useful)
#     D — no significant difference anywhere    →  no discriminative power
#
# WHY IT MATTERS:
#   Without this analysis, you can't distinguish between "this feature is high in PD"
#   and "this feature is high in everyone over 60". Pattern A features are the ones
#   you actually want for a disease-specific biomarker.
#   Most PD speech papers only have PD vs one HC group — they can't do this.
#   Your 3-group design makes this analysis uniquely possible.
#
# OUTPUT FILES:
#   exp3/age_confound_analysis.csv      — full table: p-values + pattern per feature×task
#   exp3/disease_specific_features.csv — Pattern A rows only
#   exp3/age_confound_heatmap.png       — −log10(p) heatmaps for contrasts (1) and (3)
#   exp3/feature_classification.png    — grouped bar chart of A/B/C/D counts per category
# =============================================================================

print("\n" + "─" * 70)
print("EXP 3 — Age confound analysis (3-group)")
print("─" * 70)

all_feats = (PHONATORY_FEATURES + PROSODIC_FEATURES + ARTICULATORY_FEATURES)
all_feats = [f for f in all_feats if f in df_all.columns]

age_records = []

for task in valid_tasks:
    df_task  = df_all[df_all['task'] == task]
    pd_data  = {f: df_task[df_task['group'] == 'parkinsonPD'][f].dropna() for f in all_feats}
    eld_data = {f: df_task[df_task['group'] == 'elderlyHC'][f].dropna()   for f in all_feats}
    yng_data = {f: df_task[df_task['group'] == 'youngHC'][f].dropna()     for f in all_feats}

    for feat in all_feats:
        pd_, eld_, yng_ = pd_data[feat], eld_data[feat], yng_data[feat]
        if len(pd_) < 2 or len(eld_) < 2 or len(yng_) < 2:
            continue

        _, p_pd_eld  = mannwhitneyu(pd_,  eld_, alternative='two-sided')
        _, p_pd_yng  = mannwhitneyu(pd_,  yng_, alternative='two-sided')
        _, p_eld_yng = mannwhitneyu(eld_, yng_, alternative='two-sided')

        d_pd_eld  = cohens_d(pd_.values, eld_.values)
        d_eld_yng = cohens_d(eld_.values, yng_.values)

        sig_pd_eld  = p_pd_eld  < 0.05
        sig_eld_yng = p_eld_yng < 0.05

        if sig_pd_eld and not sig_eld_yng:
            pattern = 'A: disease-specific'
        elif sig_pd_eld and sig_eld_yng:
            pattern = 'B: ageing + disease'
        elif (not sig_pd_eld) and sig_eld_yng:
            pattern = 'C: ageing only'
        else:
            pattern = 'D: no effect'

        age_records.append({
            'Task'                   : task,
            'Feature'                : feat.replace('_proxy', '').upper(),
            'Category'               : CATEGORY_MAP.get(feat, 'Unknown'),
            'p_PD_vs_elderlyHC'      : round(p_pd_eld,  4),
            'p_PD_vs_youngHC'        : round(p_pd_yng,   4),
            'p_elderlyHC_vs_youngHC' : round(p_eld_yng, 4),
            'd_PD_vs_elderlyHC'      : round(d_pd_eld,  3),
            'd_elderlyHC_vs_youngHC' : round(d_eld_yng, 3),
            'Pattern'                : pattern,
        })

age_df = pd.DataFrame(age_records)
age_df.to_csv(os.path.join(OUT_DIR, 'exp3', 'age_confound_analysis.csv'), index=False)
print("  Saved → {OUT_DIR}\\exp3\\age_confound_analysis.csv")

# ── summary counts ────────────────────────────────────────────────────────────
for pat in ['A: disease-specific', 'B: ageing + disease',
            'C: ageing only', 'D: no effect']:
    n = (age_df['Pattern'] == pat).sum()
    label = pat.split(': ')[1]
    print(f"  Pattern {pat[0]} ({label:19s}): {n:4d} feature×task pairs")

disease_specific = age_df[age_df['Pattern'] == 'A: disease-specific']
disease_specific.to_csv(os.path.join(OUT_DIR, 'exp3', 'disease_specific_features.csv'), index=False)
print("  Saved → {OUT_DIR}\\exp3\\disease_specific_features.csv")

# ── top disease-specific features ────────────────────────────────────────────
if not disease_specific.empty:
    top_dis = (disease_specific.groupby('Feature')['d_PD_vs_elderlyHC']
               .mean().sort_values(ascending=False).head(10))
    print("\n  Top disease-specific features (mean Cohen's d, PD vs elderlyHC):")
    for feat, d in top_dis.items():
        print(f"    {feat:25s}  d = {d:.3f}")

# ── Figure 1 — Side-by-side −log10(p) heatmaps ───────────────────────────────
if not age_df.empty:
    pivot_eld = age_df.pivot_table(
        values='p_PD_vs_elderlyHC', index='Feature', columns='Task', aggfunc='first'
    ).fillna(1.0)
    pivot_age = age_df.pivot_table(
        values='p_elderlyHC_vs_youngHC', index='Feature', columns='Task', aggfunc='first'
    ).fillna(1.0)

    n_rows = max(6, len(pivot_eld) * 0.4)
    n_cols = max(14, len(pivot_eld.columns) * 1.5)
    fig, axes = plt.subplots(1, 2, figsize=(min(n_cols, 28), n_rows))

    sns.heatmap(
        -np.log10(pivot_eld.clip(lower=1e-4)),
        annot=True, fmt='.1f', cmap='Reds', ax=axes[0],
        cbar_kws={'label': '−log₁₀(p)  PD vs elderlyHC'},
        linewidths=0.3,
    )
    axes[0].set_title('Disease contrast\n(PD vs elderlyHC)', fontweight='bold')
    axes[0].set_xlabel('')

    sns.heatmap(
        -np.log10(pivot_age.clip(lower=1e-4)),
        annot=True, fmt='.1f', cmap='Blues', ax=axes[1],
        cbar_kws={'label': '−log₁₀(p)  elderlyHC vs youngHC'},
        linewidths=0.3,
    )
    axes[1].set_title('Ageing contrast\n(elderlyHC vs youngHC)', fontweight='bold')
    axes[1].set_xlabel('')

    plt.suptitle(
        'Exp 3 — Separating disease effect from normal ageing\n'
        'Pattern A (disease-specific): high left, low right',
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'exp3', 'age_confound_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved → {OUT_DIR}\\exp3\\age_confound_heatmap.png")

# ── Figure 2 — Pattern distribution bar chart by category ────────────────────
if not age_df.empty:
    pattern_colors = {
        'A: disease-specific': '#E07B54',
        'B: ageing + disease' : '#5B8DB8',
        'C: ageing only'      : '#6DBF8A',
        'D: no effect'        : '#cccccc',
    }
    pattern_counts = (age_df.groupby(['Category', 'Pattern'])
                       .size().reset_index(name='count'))
    cats     = pattern_counts['Category'].unique()
    patterns = list(pattern_colors.keys())
    x = np.arange(len(cats))
    w = 0.18

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, pat in enumerate(patterns):
        vals = [
            pattern_counts[
                (pattern_counts['Category'] == c) &
                (pattern_counts['Pattern'] == pat)
            ]['count'].sum()
            for c in cats
        ]
        ax.bar(x + i * w, vals, w, label=pat,
               color=pattern_colors[pat], edgecolor='black', linewidth=0.5)

    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels(cats, fontsize=11)
    ax.set_ylabel('Number of feature × task pairs')
    ax.set_title('Exp 3 — Feature patterns by category\n'
                 'Pattern A = disease-specific (the ones you want)')
    ax.legend(fontsize=8, loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'exp3', 'feature_classification.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved → {OUT_DIR}\\exp3\\feature_classification.png")

print("✅ Exp 3 complete → {OUT_DIR}\\exp3\\")


# =============================================================================
# EXP 4 — TASK-TYPE SENSITIVITY
# =============================================================================
# WHAT IT DOES:
#   Trains an RF and an SVM separately for every combination of task × feature category.
#   Uses StratifiedGroupKFold (speakers never leak between train and test).
#   Records accuracy and std for each combination, then produces:
#     - a table: which task is best for each category?
#     - a heatmap: RF accuracy per task (rows) × category (columns)
#
# WHY IT MATTERS:
#   Jitter and shimmer come from glottal instability — they are meaningful on sustained
#   vowel tasks but essentially undefined during rapid consonant sequences.
#   F0 dynamics (df0, ddf0) require pitch movement — they are meaningful in connected
#   speech or diadochokinesis but flat in a sustained /a/.
#   If you train on all features regardless of task type, the model uses features
#   that are physiologically irrelevant for that task.
#   This experiment shows which (task, category) combinations actually work, and
#   lets you make a clinical recommendation: "to screen for PD, use THIS task with
#   THESE features and you achieve X% accuracy with just Y features".
#
# OUTPUT FILES:
#   exp4/task_category_accuracy.csv   — full table: accuracy per task × category × model
#   exp4/best_task_per_category.csv   — best task per category (RF only)
#   exp4/task_type_heatmap.png        — heatmap of RF accuracy
#   exp4/model_comparison.png         — RF vs SVM per category (aggregated across tasks)
# =============================================================================

print("\n" + "─" * 70)
print("EXP 4 — Task-type sensitivity")
print("─" * 70)

MODELS_4 = [
    ('RF',  Pipeline([('rf', RandomForestClassifier(n_estimators=200, random_state=42))])),
    ('SVM', Pipeline([('sc', StandardScaler()), ('svm', SVC(random_state=42))])),
]

task4_records = []

for task in valid_tasks:
    df_task = df_all[df_all['task'] == task].copy()

    for cat, cat_feats in [
        ('Phonatory',    PHONATORY_FEATURES),
        ('Prosodic',     PROSODIC_FEATURES),
        ('Articulatory', ARTICULATORY_FEATURES),
    ]:
        avail = [f for f in cat_feats
                 if f in df_task.columns and df_task[f].notna().any()]
        if len(avail) < 2:
            continue

        X, y, groups = prep_X(df_task, avail)
        if y.nunique() < 2 or len(y) < 10:
            continue

        cv_res = run_cv(X, y, groups, MODELS_4)
        for name, accs in cv_res.items():
            if accs:
                task4_records.append({
                    'Task'      : task,
                    'Category'  : cat,
                    'Model'     : name,
                    'Accuracy'  : round(np.mean(accs), 3),
                    'Std'       : round(np.std(accs),  3),
                    'n_features': len(avail),
                })

task4_df = pd.DataFrame(task4_records)
task4_df.to_csv(os.path.join(OUT_DIR, 'exp4', 'task_category_accuracy.csv'), index=False)
print("  Saved → {OUT_DIR}\\exp4\\task_category_accuracy.csv")

if task4_df.empty:
    print("  ⚠️  No results — check that feature columns exist in your CSV.")
else:
    # ── Best task per category (RF) ───────────────────────────────────────────
    best = (task4_df[task4_df['Model'] == 'RF']
            .sort_values('Accuracy', ascending=False)
            .groupby('Category').first()
            .reset_index()[['Category', 'Task', 'Accuracy', 'Std', 'n_features']])
    best.to_csv(os.path.join(OUT_DIR, 'exp4', 'best_task_per_category.csv'), index=False)
    print("  Saved → {OUT_DIR}\\exp4\\best_task_per_category.csv")
    print("\n  Best task per feature category (RF):")
    print(best.to_string(index=False))

    # ── Figure 1 — RF accuracy heatmap: task × category ──────────────────────
    pivot4 = (task4_df[task4_df['Model'] == 'RF']
              .pivot_table(index='Task', columns='Category',
                           values='Accuracy', aggfunc='mean')
              .fillna(0))

    fig, ax = plt.subplots(figsize=(
        max(6, len(pivot4.columns) * 2.5),
        max(5, len(pivot4) * 0.55),
    ))
    sns.heatmap(
        pivot4, annot=True, fmt='.3f', cmap='YlOrRd',
        vmin=0.4, vmax=1.0, ax=ax, linewidths=0.4,
        cbar_kws={'label': 'RF accuracy (speaker-split CV)'},
    )
    ax.set_title(
        'Exp 4 — RF accuracy per task × feature category\n'
        'High cell = this task + this feature type discriminates PD well',
        fontweight='bold',
    )
    ax.set_xlabel('Feature category')
    ax.set_ylabel('Task')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'exp4', 'task_type_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved → {OUT_DIR}\\exp4\\task_type_heatmap.png")

    # ── Figure 2 — RF vs SVM per category (mean accuracy across all tasks) ───
    model_agg = (task4_df.groupby(['Category', 'Model'])['Accuracy']
                 .agg(['mean', 'std']).reset_index())
    categories = model_agg['Category'].unique()
    x = np.arange(len(categories))
    w = 0.3

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (model_name, color) in enumerate([('RF', '#4A90D9'), ('SVM', '#E07B54')]):
        sub = model_agg[model_agg['Model'] == model_name]
        # align to category order
        vals = [sub[sub['Category'] == c]['mean'].values[0]
                if len(sub[sub['Category'] == c]) > 0 else 0
                for c in categories]
        errs = [sub[sub['Category'] == c]['std'].values[0]
                if len(sub[sub['Category'] == c]) > 0 else 0
                for c in categories]
        bars = ax.bar(x + i * w, vals, w, label=model_name,
                      color=color, edgecolor='black', linewidth=0.5, alpha=0.85)
        ax.errorbar(x + i * w, vals, errs,
                    fmt='none', color='black', capsize=3, linewidth=0.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + 0.01, f'{v:.3f}',
                    ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x + w / 2)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Mean accuracy (± std across tasks)')
    ax.set_title('Exp 4 — RF vs SVM accuracy per feature category\n'
                 '(averaged across all tasks)')
    ax.axhline(0.5, linestyle='--', color='gray', linewidth=0.7, label='Chance (0.5)')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'exp4', 'model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved → {OUT_DIR}\\exp4\\model_comparison.png")

print("✅ Exp 4 complete → {OUT_DIR}\\exp4\\")


# =============================================================================
# ── SUMMARY ───────────────────────────────────────────────────────────────────
# =============================================================================

print("\n" + "=" * 70)
print("EXPERIMENTS 3 & 4 COMPLETE")
print("=" * 70)
print("""
Exp 3  {OUT_DIR}\\exp3\\
       age_confound_analysis.csv      ← 3-way p-values + A/B/C/D pattern per feature×task
       disease_specific_features.csv  ← Pattern A rows only (disease-specific markers)
       age_confound_heatmap.png       ← side-by-side heatmaps, disease vs ageing contrast
       feature_classification.png    ← grouped bar chart of pattern counts per category

Exp 4  {OUT_DIR}\\exp4\\
       task_category_accuracy.csv    ← RF + SVM accuracy per task × category
       best_task_per_category.csv    ← which task is best for each category
       task_type_heatmap.png         ← heatmap of RF accuracy (tasks × categories)
       model_comparison.png          ← RF vs SVM grouped bar chart

NEXT STEP: run pd_exp5_exp6.py
""")