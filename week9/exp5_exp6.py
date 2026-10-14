"""
=============================================================================
PD SPEECH — EXPERIMENTS 5 & 6
Run AFTER your original pipeline. Can run standalone; does not need outputs
from pd_exp1_exp2.py (but will use articulatory features if you add them below).

EXP 5 — SHAP explainability
  Answers: "How exactly does the model weight each feature? Are they all equal?"
  Trains an RF on all tasks combined, then uses SHAP TreeExplainer to compute
  per-sample, per-feature contributions. Produces a beeswarm plot (direction +
  magnitude) and a category-level summary (which category the model relies on).
  Requires: pip install shap
  OUTPUT → results_week9/exp5/

EXP 6 — Late-fusion model
  Answers: "Do the feature categories capture DIFFERENT information?"
  Trains a separate RF for each category, collects class probabilities, then
  averages them (late fusion). If FUSION > best single category, the categories
  are complementary — which is the core claim of the 3-category framework.
  Also adds AUC-ROC (not just accuracy) for a more informative comparison.
  OUTPUT → results_week9/exp6/

HOW TO RUN:
  pip install shap    # one-time
  python exp5_exp6.py
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
from sklearn.metrics import accuracy_score, roc_auc_score

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
# If you ran exp1_exp2.py and Exp 2 extracted articulatory features,
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

OUT_DIR = r'D:\Download\Codes\DH604\week9\results_week9'
for folder in [os.path.join(OUT_DIR, 'exp5'), os.path.join(OUT_DIR, 'exp6')]:
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
print("PD SPEECH — EXPERIMENTS 5 & 6")
print("=" * 70)
print(f"  Dataset : {df_all.shape[0]} rows, {len(valid_tasks)} tasks")
print(f"  Groups  : {df_all['group'].value_counts().to_dict()}")


# =============================================================================
# EXP 5 — SHAP EXPLAINABILITY
# =============================================================================

print("\n" + "─" * 70)
print("EXP 5 — SHAP explainability")
print("─" * 70)

try:
    import shap
    SHAP_OK = True
except ImportError:
    SHAP_OK = False
    print("  ⚠️  shap not installed.  Run:  pip install shap")
    print("      Skipping Exp 5.")

if SHAP_OK:
    all_feats = PHONATORY_FEATURES + PROSODIC_FEATURES + ARTICULATORY_FEATURES
    all_feats = [f for f in all_feats if f in df_all.columns]

    sub5   = df_all[all_feats + ['label', 'speaker']].dropna(subset=['label'])
    avail5 = [f for f in all_feats if sub5[f].notna().any()]

    imp5 = SimpleImputer(strategy='median')
    X5   = pd.DataFrame(imp5.fit_transform(sub5[avail5]),
                        columns=avail5, index=sub5.index)
    y5   = sub5['label']

    if y5.nunique() < 2:
        print("  ⚠️  Only one class in data — cannot run SHAP.")
    elif len(y5) < 20:
        print("  ⚠️  Too few samples (need ≥ 20) — cannot run SHAP.")
    else:
        print(f"  Training RF on {len(y5)} samples, {len(avail5)} features ...")
        rf5 = RandomForestClassifier(n_estimators=300, random_state=42)
        rf5.fit(X5, y5)
        print(f"  Training accuracy: {rf5.score(X5, y5):.3f}  "
              f"(this is in-sample; cross-validation is in Exp 6)")

        print("  Computing SHAP values (TreeExplainer) ...")
        explainer = shap.TreeExplainer(rf5)
        shap_vals = explainer.shap_values(X5)

        # Handle all shap output formats across library versions:
        #   old shap (<0.40) : list of two 2D arrays  → take [1] for PD class
        #   new shap (>=0.40): single 3D array (n_samples, n_features, n_classes) → take [:,:,1]
        if isinstance(shap_vals, list):
            sv = shap_vals[1]
        elif hasattr(shap_vals, 'ndim') and shap_vals.ndim == 3:
            sv = shap_vals[:, :, 1]
        else:
            sv = shap_vals

        shap_df = pd.DataFrame(sv, columns=avail5, index=X5.index)
        shap_df.to_csv(os.path.join(OUT_DIR, 'exp5', 'shap_values.csv'), index=False)
        print(f"  Saved → {OUT_DIR}\\exp5\\shap_values.csv")

        mean_abs = pd.DataFrame({
            'Feature'        : avail5,
            'Mean_abs_SHAP'  : np.abs(sv).mean(axis=0),
            'Category'       : [CATEGORY_MAP.get(f, 'Unknown') for f in avail5],
        }).sort_values('Mean_abs_SHAP', ascending=False).reset_index(drop=True)
        mean_abs['Feature_label'] = (mean_abs['Feature']
                                     .str.replace('_proxy', '').str.upper())
        mean_abs.to_csv(os.path.join(OUT_DIR, 'exp5', 'shap_mean_abs.csv'), index=False)
        print(f"  Saved → {OUT_DIR}\\exp5\\shap_mean_abs.csv")
        print(f"\n  Top-5 features by mean |SHAP| (positive = pushes toward PD):")
        print(mean_abs.head(5)[['Feature_label', 'Category', 'Mean_abs_SHAP']]
              .to_string(index=False))

        # ── Beeswarm plot ─────────────────────────────────────────────────────
        top_n     = min(15, len(avail5))
        top_feats = mean_abs.head(top_n)['Feature'].tolist()
        sv_top    = sv[:, [avail5.index(f) for f in top_feats]]
        X5_top    = X5[top_feats].values
        X5_norm   = (X5_top - X5_top.min(axis=0)) / (np.ptp(X5_top, axis=0) + 1e-9)

        fig, ax = plt.subplots(figsize=(11, max(6, top_n * 0.45)))
        sc = None
        for i, feat in enumerate(top_feats):
            x_vals   = sv_top[:, i]
            c_vals   = X5_norm[:, i]
            y_jitter = np.random.default_rng(i).uniform(-0.32, 0.32, len(x_vals))
            sc = ax.scatter(x_vals, i + y_jitter, c=c_vals,
                            cmap='coolwarm', alpha=0.5, s=14, vmin=0, vmax=1)

        ax.axvline(0, color='black', linewidth=1.0, linestyle='--')
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([f.replace('_proxy', '').upper() for f in top_feats], fontsize=9)
        ax.set_xlabel('SHAP value  (positive → model predicts PD,  negative → model predicts HC)')
        ax.set_title(
            f'Exp 5 — SHAP beeswarm: how each feature influences the model\n'
            f'Each dot = one row. Colour: red = high value, blue = low value. Top {top_n} features shown.',
            fontsize=10)

        if sc is not None:
            cbar = plt.colorbar(sc, ax=ax, shrink=0.45, pad=0.01)
            cbar.set_label('Feature value (normalised)', fontsize=8)

        for i, feat in enumerate(top_feats):
            cat = CATEGORY_MAP.get(feat, 'Unknown')
            ax.add_patch(mpatches.Rectangle(
                (-0.075, i - 0.45), 0.045, 0.9,
                transform=ax.get_yaxis_transform(),
                color=CATEGORY_COLORS.get(cat, '#ccc'), clip_on=False))

        legend_patches = [mpatches.Patch(color=v, label=k)
                          for k, v in CATEGORY_COLORS.items()
                          if k in mean_abs['Category'].values]
        ax.legend(handles=legend_patches, loc='lower right', fontsize=8, title='Category')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'exp5', 'shap_beeswarm.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved → {OUT_DIR}\\exp5\\shap_beeswarm.png")

        # ── Category-level SHAP summary ───────────────────────────────────────
        cat_shap = (mean_abs.groupby('Category')['Mean_abs_SHAP']
                    .sum().reset_index()
                    .sort_values('Mean_abs_SHAP', ascending=False))
        fig, ax = plt.subplots(figsize=(7, 4))
        colors_s = [CATEGORY_COLORS.get(c, '#aaa') for c in cat_shap['Category']]
        ax.bar(cat_shap['Category'], cat_shap['Mean_abs_SHAP'],
               color=colors_s, edgecolor='black', linewidth=0.6)
        ax.set_ylabel('Sum of mean |SHAP| across all features in category')
        ax.set_title('Exp 5 — Which feature category does the model rely on most?\n'
                     'Answers: "are all features weighted equally?" — No.')
        for i, row in enumerate(cat_shap.itertuples()):
            ax.text(i, row.Mean_abs_SHAP * 1.02,
                    f'{row.Mean_abs_SHAP:.4f}', ha='center', fontsize=10, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'exp5', 'shap_category_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved → {OUT_DIR}\\exp5\\shap_category_summary.png")

    print(f"✅ Exp 5 complete → {OUT_DIR}\\exp5\\")
else:
    print("  Exp 5 skipped.")


# =============================================================================
# EXP 6 — LATE-FUSION MODEL
# =============================================================================

print("\n" + "─" * 70)
print("EXP 6 — Late-fusion model")
print("─" * 70)

fusion_records = []

for task in valid_tasks:
    df_task = df_all[df_all['task'] == task].copy()
    y_all   = df_task['label']
    grp_all = df_task['speaker']

    if y_all.nunique() < 2 or len(y_all) < 10:
        continue
    n_splits = min(5, int(y_all.value_counts().min()))
    if n_splits < 2:
        continue

    cat_feat_map = {
        cat: [f for f in feats if f in df_task.columns]
        for cat, feats in [
            ('Phonatory',    PHONATORY_FEATURES),
            ('Prosodic',     PROSODIC_FEATURES),
            ('Articulatory', ARTICULATORY_FEATURES),
        ]
    }
    cat_feat_map = {k: v for k, v in cat_feat_map.items() if len(v) >= 2}
    if len(cat_feat_map) < 2:
        continue

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_accs_fusion = []
    fold_aucs_fusion = []
    fold_accs_single = {cat: [] for cat in cat_feat_map}
    fold_aucs_single = {cat: [] for cat in cat_feat_map}

    for tr_idx, te_idx in sgkf.split(df_task, y_all, grp_all):
        if len(np.unique(y_all.iloc[tr_idx])) < 2:
            continue
        cat_probas_te = []
        for cat, feats in cat_feat_map.items():
            imp  = SimpleImputer(strategy='median')
            X_tr = pd.DataFrame(imp.fit_transform(df_task.iloc[tr_idx][feats]), columns=feats)
            X_te = pd.DataFrame(imp.transform(df_task.iloc[te_idx][feats]), columns=feats)
            y_tr = y_all.iloc[tr_idx]
            y_te = y_all.iloc[te_idx]
            rf   = RandomForestClassifier(n_estimators=200, random_state=42)
            rf.fit(X_tr, y_tr)
            proba_te = rf.predict_proba(X_te)[:, 1]
            pred_te  = rf.predict(X_te)
            fold_accs_single[cat].append(accuracy_score(y_te, pred_te))
            if y_te.nunique() == 2:
                fold_aucs_single[cat].append(roc_auc_score(y_te, proba_te))
            cat_probas_te.append(proba_te)

        if len(cat_probas_te) >= 2:
            fused_proba = np.mean(cat_probas_te, axis=0)
            fused_pred  = (fused_proba >= 0.5).astype(int)
            y_te_fused  = y_all.iloc[te_idx]
            fold_accs_fusion.append(accuracy_score(y_te_fused, fused_pred))
            if y_te_fused.nunique() == 2:
                fold_aucs_fusion.append(roc_auc_score(y_te_fused, fused_proba))

    if fold_accs_fusion:
        fusion_records.append({
            'Task': task, 'Category': 'FUSION',
            'Accuracy': round(np.mean(fold_accs_fusion), 3),
            'Acc_Std' : round(np.std(fold_accs_fusion),  3),
            'AUC'     : round(np.mean(fold_aucs_fusion), 3) if fold_aucs_fusion else None,
            'AUC_Std' : round(np.std(fold_aucs_fusion),  3) if fold_aucs_fusion else None,
        })
        auc_str = f"AUC={np.mean(fold_aucs_fusion):.3f}" if fold_aucs_fusion else 'AUC=N/A'
        print(f"  {task:30s} | FUSION     : "
              f"Acc={np.mean(fold_accs_fusion):.3f}±{np.std(fold_accs_fusion):.3f}  {auc_str}")

    for cat, accs in fold_accs_single.items():
        if accs:
            aucs = fold_aucs_single[cat]
            fusion_records.append({
                'Task': task, 'Category': cat,
                'Accuracy': round(np.mean(accs), 3),
                'Acc_Std' : round(np.std(accs),  3),
                'AUC'     : round(np.mean(aucs), 3) if aucs else None,
                'AUC_Std' : round(np.std(aucs),  3) if aucs else None,
            })
            auc_str = f"AUC={np.mean(aucs):.3f}" if aucs else 'AUC=N/A'
            print(f"  {task:30s} | {cat:12s}: "
                  f"Acc={np.mean(accs):.3f}±{np.std(accs):.3f}  {auc_str}")

fusion_df = pd.DataFrame(fusion_records)
fusion_df.to_csv(os.path.join(OUT_DIR, 'exp6', 'fusion_results.csv'), index=False)
print(f"  Saved → {OUT_DIR}\\exp6\\fusion_results.csv")

if fusion_df.empty:
    print("  ⚠️  No fusion results — need at least 2 feature categories with ≥ 2 features.")
else:
    agg = (fusion_df.groupby('Category')
           .agg(Accuracy=('Accuracy', 'mean'), Acc_Std=('Acc_Std', 'mean'), AUC=('AUC', 'mean'))
           .reset_index()
           .sort_values('Accuracy', ascending=False))

    print("\n  Overall mean accuracy per strategy:")
    print(agg[['Category', 'Accuracy', 'Acc_Std', 'AUC']].to_string(index=False))

    # ── Figure 1 — Accuracy bar chart ────────────────────────────────────────
    bar_colors = ['#2C2C2A' if c == 'FUSION' else CATEGORY_COLORS.get(c, '#aaa')
                  for c in agg['Category']]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(agg['Category'], agg['Accuracy'],
                  color=bar_colors, edgecolor='black', linewidth=0.6)
    ax.errorbar(range(len(agg)), agg['Accuracy'], agg['Acc_Std'],
                fmt='none', color='black', capsize=4, linewidth=1.0)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('Mean accuracy (speaker-split CV)')
    ax.set_title('Exp 6 — Late-fusion vs single-category classifiers\n'
                 'FUSION > all single categories → categories are complementary')
    ax.axhline(0.5, linestyle='--', color='gray', linewidth=0.7, label='Chance (0.5)')
    for bar, row in zip(bars, agg.itertuples()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f'{row.Accuracy:.3f}', ha='center', fontsize=10, fontweight='bold')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'exp6', 'fusion_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {OUT_DIR}\\exp6\\fusion_comparison.png")

    # ── Figure 2 — AUC-ROC ───────────────────────────────────────────────────
    agg_auc = agg.dropna(subset=['AUC']).sort_values('AUC', ascending=False)
    if not agg_auc.empty:
        bar_colors_auc = ['#2C2C2A' if c == 'FUSION' else CATEGORY_COLORS.get(c, '#aaa')
                          for c in agg_auc['Category']]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(agg_auc['Category'], agg_auc['AUC'],
               color=bar_colors_auc, edgecolor='black', linewidth=0.6)
        ax.set_ylim(0, 1.12)
        ax.axhline(0.5, linestyle='--', color='gray', linewidth=0.7, label='Chance (0.5)')
        ax.axhline(0.8, linestyle='--', color='steelblue', linewidth=0.7, label='Good (0.8)')
        ax.set_ylabel('Mean AUC-ROC (speaker-split CV)')
        ax.set_title('Exp 6 — AUC-ROC: FUSION vs single categories')
        for i, row in enumerate(agg_auc.itertuples()):
            ax.text(i, row.AUC + 0.02, f'{row.AUC:.3f}',
                    ha='center', fontsize=10, fontweight='bold')
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'exp6', 'fusion_auc.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved → {OUT_DIR}\\exp6\\fusion_auc.png")

    # ── Figure 3 — Per-task heatmap ───────────────────────────────────────────
    pivot6 = fusion_df.pivot_table(
        index='Task', columns='Category', values='Accuracy', aggfunc='mean').fillna(0)
    if not pivot6.empty:
        fig, ax = plt.subplots(figsize=(max(6, len(pivot6.columns)*2), max(5, len(pivot6)*0.55)))
        sns.heatmap(pivot6, annot=True, fmt='.3f', cmap='YlOrRd',
                    vmin=0.4, vmax=1.0, ax=ax, linewidths=0.3,
                    cbar_kws={'label': 'Accuracy'})
        ax.set_title('Exp 6 — Accuracy per task × strategy (FUSION included)', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'exp6', 'fusion_task_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved → {OUT_DIR}\\exp6\\fusion_task_heatmap.png")

print(f"✅ Exp 6 complete → {OUT_DIR}\\exp6\\")


# =============================================================================
# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
# =============================================================================

print("\n" + "=" * 70)
print("EXPERIMENTS 5 & 6 COMPLETE — ALL 6 EXPERIMENTS DONE")
print("=" * 70)
print(f"""
Exp 5  {OUT_DIR}\\exp5\\
       shap_values.csv           ← raw SHAP per sample × feature
       shap_mean_abs.csv         ← mean |SHAP| per feature, ranked + categorised
       shap_beeswarm.png         ← publication-ready beeswarm (direction + magnitude)
       shap_category_summary.png ← which category the model relies on most

Exp 6  {OUT_DIR}\\exp6\\
       fusion_results.csv        ← accuracy + AUC per task × category + FUSION
       fusion_comparison.png     ← accuracy bar chart (key result figure)
       fusion_auc.png            ← AUC-ROC comparison
       fusion_task_heatmap.png   ← per-task accuracy heatmap
""")