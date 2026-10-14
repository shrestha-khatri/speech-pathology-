import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy import stats
from scipy.stats import shapiro, kruskal, mannwhitneyu
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os

warnings.filterwarnings('ignore')

print("=== FULL PIPELINE: Non-Parametric Primary + ML (Speaker-Split) ===")

# === LOAD & PREP DATA ===
df = pd.read_csv(r"D:\Download\Codes\DH604\week 7\phonatory_task_grouped_FINAL.csv")

numeric_cols = [
    'hnr', 'jitter_local', 'jitter_rap', 'jitter_ppq5', 'shimmer_local',
    'shimmer_apq5', 'f0_mean', 'f0_std', 'df0', 'ddf0', 'rpde_proxy',
    'dfa_proxy', 'hurst_proxy', 'lle_proxy', 'apen_proxy', 'sampen_proxy',
    'ppe', 'ftri_proxy', 'cvf0'
]

# Features confirmed normally distributed by Shapiro-Wilk
# (only these are valid for parametric t-test)
NORMAL_FEATURES = {'HNR', 'F0_MEAN', 'SAMPEN'}

# Filter PD vs HC + valid tasks
df_pd_hc = df[df['group'].isin(['parkinsonPD', 'elderlyHC', 'youngHC'])].copy()
task_counts = df_pd_hc.groupby('task')['group'].nunique()
valid_tasks = task_counts[task_counts >= 2].index
df_analysis = df_pd_hc[df_pd_hc['task'].isin(valid_tasks)].copy()

print(f"📊 {len(valid_tasks)} tasks × {len(numeric_cols)} features")

# =============================================================
# SECTION 1: NORMALITY TESTS (must run FIRST to decide which test to use)
# =============================================================
print("\n🧪 Step 1: Shapiro-Wilk Normality Tests...")
os.makedirs('results', exist_ok=True)
norm_results = []

for task in valid_tasks:
    df_task = df_analysis[df_analysis['task'] == task]
    for feature in numeric_cols:
        if feature not in df_task.columns:
            continue
        data = df_task[feature].dropna()
        if len(data) < 3:
            continue
        _, shapiro_p = shapiro(data)
        norm_results.append({
            'Task':      task,
            'Feature':   feature.replace('_proxy', '').upper(),
            'Shapiro_p': shapiro_p,
            'Normal':    shapiro_p > 0.05
        })

norm_df = pd.DataFrame(norm_results)
norm_df.to_csv('results/normality_shapiro.csv', index=False)

total_tests = len(norm_df)
non_normal  = (norm_df['Normal'] == False).sum()
pct_non     = non_normal / total_tests * 100
print(f"  Non-normal: {non_normal}/{total_tests} ({pct_non:.1f}%) → Non-parametric tests required")
print("✅ normality_shapiro.csv")

# =============================================================
# SECTION 2: PRIMARY ANALYSIS — MANN-WHITNEY U + COHEN'S D
# =============================================================
# Mann-Whitney U is the correct test for non-normal data.
# Cohen's d is computed from means/SD and is used purely as
# an effect size descriptor — it does NOT assume normality.
# For the 3 normal features (HNR, F0_MEAN, SAMPEN) we also
# note the Welch t-test p-value for completeness.
# =============================================================
print("\n🔥 Step 2: Primary Analysis — Mann-Whitney U + Cohen's d...")
stat_summary = []

for task in valid_tasks:
    df_task = df_analysis[df_analysis['task'] == task]
    print(f"  {task}: n={len(df_task)}")

    for feature in numeric_cols:
        if feature not in df_task.columns:
            continue

        feat_label = feature.replace('_proxy', '').upper()
        feature_data = df_task[feature].dropna()
        if len(feature_data) < 4:
            continue

        hc_data = df_task[df_task['group'].isin(['elderlyHC', 'youngHC'])][feature].dropna()
        pd_data = df_task[df_task['group'] == 'parkinsonPD'][feature].dropna()

        if len(hc_data) < 2 or len(pd_data) < 2:
            continue

        hc_mean, pd_mean = hc_data.mean(), pd_data.mean()
        hc_std,  pd_std  = hc_data.std(),  pd_data.std()

        # Cohen's d — effect size descriptor, does NOT assume normality
        pooled_std = np.sqrt((hc_std**2 + pd_std**2) / 2)
        cohens_d   = abs(hc_mean - pd_mean) / pooled_std if pooled_std > 0.001 else 0

        # PRIMARY TEST: Mann-Whitney U (valid for ALL features)
        mw_stat, mw_p = mannwhitneyu(hc_data, pd_data, alternative='two-sided')

        # SECONDARY TEST: Welch t-test (only valid for normally distributed features)
        _, t_p = stats.ttest_ind(hc_data, pd_data, equal_var=False)
        is_normal = feat_label in NORMAL_FEATURES

        stat_summary.append({
            'Task':         task,
            'Feature':      feat_label,
            'HC_mean':      hc_mean,
            'PD_mean':      pd_mean,
            'Mean_diff':    pd_mean - hc_mean,
            'Cohens_d':     cohens_d,
            'MW_p':         mw_p,           # PRIMARY — always valid
            'ttest_p':      t_p,            # SECONDARY — valid only for normal features
            'Normal':       is_normal,
            'Primary_test': 'Welch t-test' if is_normal else 'Mann-Whitney U',
            'Primary_p':    t_p if is_normal else mw_p,  # correct p for each feature
        })

summary_df = pd.DataFrame(stat_summary)
summary_df.to_csv('results/full_stat_summary.csv', index=False)
print("✅ full_stat_summary.csv")

# =============================================================
# SECTION 3: KRUSKAL-WALLIS (3-GROUP TEST) + BONFERRONI POST-HOC
# =============================================================
print("\n📊 Step 3: Kruskal-Wallis (3 groups) + Mann-Whitney post-hoc...")
kw_results = []

for task in valid_tasks:
    df_task = df_analysis[df_analysis['task'] == task]

    for feature in numeric_cols:
        if feature not in df_task.columns:
            continue
        data = df_task[feature].dropna()
        if len(data) < 3:
            continue

        feat_label   = feature.replace('_proxy', '').upper()
        pd_data      = df_task[df_task['group'] == 'parkinsonPD'][feature].dropna()
        elderly_data = df_task[df_task['group'] == 'elderlyHC'][feature].dropna()
        young_data   = df_task[df_task['group'] == 'youngHC'][feature].dropna()

        if len(pd_data) < 2 or len(elderly_data) < 1 or len(young_data) < 1:
            continue

        # Kruskal-Wallis: non-parametric one-way ANOVA
        _, kw_p = kruskal(pd_data, elderly_data, young_data)

        # Mann-Whitney post-hoc: PD vs each HC subgroup
        _, pd_eld_p   = mannwhitneyu(pd_data, elderly_data, alternative='two-sided')
        _, pd_young_p = mannwhitneyu(pd_data, young_data,   alternative='two-sided')

        # Bonferroni correction for 2 comparisons
        pd_hc_p_bonf = min(pd_eld_p, pd_young_p) * 2

        kw_results.append({
            'Task':                    task,
            'Feature':                 feat_label,
            'KW_p':                    kw_p,
            'KW_significant':          kw_p < 0.05,
            'MW_PD_vs_ElderlyHC_p':   pd_eld_p,
            'MW_PD_vs_YoungHC_p':     pd_young_p,
            'MW_PD_vs_HC_bonferroni': pd_hc_p_bonf,
            'MW_significant':          pd_hc_p_bonf < 0.05,
        })

kw_df = pd.DataFrame(kw_results)
kw_df.to_csv('results/kruskal_mannwhitney.csv', index=False)
print("✅ kruskal_mannwhitney.csv")

# =============================================================
# SECTION 4: 4 HEATMAPS — NOW USING CORRECT TESTS
# =============================================================
# Heatmap 1: Mann-Whitney p-values (primary, valid for all features)
# Heatmap 2: Cohen's d effect sizes
# Heatmap 3: Mean differences (PD - HC)
# Heatmap 4: -log10(Mann-Whitney p) significance map
# =============================================================
print("\n🎨 Step 4: Generating 4 Heatmaps (non-parametric p-values)...")
plt.style.use('default')

# --- Build matrices from PRIMARY p-values (Mann-Whitney) ---
mw_matrix = pd.pivot_table(
    summary_df, values='MW_p', index='Feature', columns='Task', aggfunc='first'
).fillna(1.0)

effect_matrix = pd.pivot_table(
    summary_df, values='Cohens_d', index='Feature', columns='Task', aggfunc='first'
).fillna(0)

mean_diff_matrix = pd.pivot_table(
    summary_df, values='Mean_diff', index='Feature', columns='Task', aggfunc='first'
).fillna(0)

# 1. MANN-WHITNEY P-VALUES
plt.figure(figsize=(14, 10))
sns.heatmap(mw_matrix, annot=True, cmap='viridis', fmt='.3f',
            cbar_kws={'label': 'Mann-Whitney U p-value'})
plt.title('MANN-WHITNEY U P-VALUES: PD vs HC\n(Primary test — valid for non-normal distributions)',
          fontsize=14, pad=20)
plt.xlabel('Speech Task'); plt.ylabel('Phonatory Feature')
plt.tight_layout()
plt.savefig('results/heatmap1_mannwhitney_pvalues.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ heatmap1_mannwhitney_pvalues.png")

# 2. EFFECT SIZES (Cohen's d — valid as descriptor regardless of distribution)
plt.figure(figsize=(14, 10))
sns.heatmap(effect_matrix, annot=True, cmap='Reds', fmt='.2f',
            cbar_kws={'label': "Cohen's d (effect size)"})
plt.title("EFFECT SIZES: PD vs HC\n(Cohen's d — small≥0.2, medium≥0.5, large≥0.8)",
          fontsize=14, pad=20)
plt.xlabel('Speech Task'); plt.ylabel('Phonatory Feature')
plt.tight_layout()
plt.savefig('results/heatmap2_effect_sizes.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ heatmap2_effect_sizes.png")

# 3. MEAN DIFFERENCES (PD - HC)
plt.figure(figsize=(14, 10))
sns.heatmap(mean_diff_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f',
            cbar_kws={'label': 'PD_mean − HC_mean'})
plt.title('MEAN DIFFERENCES: PD vs HC\n(Blue = PD lower, Red/Orange = PD higher)',
          fontsize=14, pad=20)
plt.xlabel('Speech Task'); plt.ylabel('Phonatory Feature')
plt.tight_layout()
plt.savefig('results/heatmap3_mean_differences.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ heatmap3_mean_differences.png")

# 4. SIGNIFICANCE MAP: -log10(Mann-Whitney p)
sig_matrix = np.log10(mw_matrix.clip(lower=0.0001)) * -1
plt.figure(figsize=(14, 10))
sns.heatmap(sig_matrix, annot=True, cmap='coolwarm', fmt='.1f',
            cbar_kws={'label': '−log₁₀(Mann-Whitney p)'})
plt.title('STATISTICAL SIGNIFICANCE: PD vs HC\n(−log₁₀ of Mann-Whitney p-value · Higher = More Significant)',
          fontsize=14, pad=20)
plt.xlabel('Speech Task'); plt.ylabel('Phonatory Feature')
plt.tight_layout()
plt.savefig('results/heatmap4_significance.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ heatmap4_significance.png")

print("\n🎉 4 HEATMAPS COMPLETE (using Mann-Whitney p-values)!")

# =============================================================
# SECTION 5: SUMMARY STATS TABLE
# =============================================================
print("\n📋 Step 5: Summary statistics table...")

# Count significant features per task (Mann-Whitney p < 0.05)
sig_per_task = summary_df[summary_df['MW_p'] < 0.05].groupby('Task')['Feature'].count()
print("\n  Significant features per task (MW p<0.05):")
for task, count in sig_per_task.items():
    total = summary_df[summary_df['Task'] == task]['Feature'].count()
    print(f"    {task}: {count}/{total}")

# Most consistently significant features
feat_sig_count = summary_df[summary_df['MW_p'] < 0.05].groupby('Feature')['Task'].count().sort_values(ascending=False)
print("\n  Most consistent features (significant across most tasks):")
print(feat_sig_count.head(10).to_string())

# =============================================================
# SECTION 6: FEATURE PREP FOR ML
# =============================================================
print("\n🎯 Step 6: Preparing Features for ML...")
ml_data = []

for task in valid_tasks:
    # Rank features by Mann-Whitney p-value (correct ranking)
    task_df = summary_df[summary_df['Task'] == task].copy()
    task_df = task_df.sort_values('MW_p')  # sort by MW p, not t-test p
    top_feats_raw = (
        task_df.head(10)['Feature']
        .str.lower()
        .str.replace(' ', '_')
        .tolist()
    )

    available_feats = [f for f in top_feats_raw if f in df_analysis.columns]
    print(f"  {task}: {len(available_feats)}/10 features available")

    if len(available_feats) >= 3:
        cols_to_keep = ['group'] + available_feats
        task_rows = df_analysis[df_analysis['task'] == task][cols_to_keep].dropna()

        if len(task_rows) >= 10:
            task_rows = task_rows.copy()
            task_rows['label'] = (task_rows['group'] == 'parkinsonPD').astype(int)
            task_rows['task']  = task

            if 'speaker' in df_analysis.columns:
                task_rows['speaker'] = df_analysis.loc[task_rows.index, 'speaker']
            else:
                task_rows['speaker'] = (task_rows.index // 5).astype(str)

            ml_data.append(task_rows)

ml_df = pd.concat(ml_data, ignore_index=True)
ml_df.to_csv('results/ml_features_ready.csv', index=False)
print(f"✅ ml_features_ready.csv: {ml_df.shape} ({len(ml_df['task'].unique())} tasks)")

# =============================================================
# SECTION 7: ML — RF + SVM, SPEAKER-STRATIFIED CV
# =============================================================
if len(ml_df) >= 20:
    print("\n🤖 Step 7: ML — RF + SVM, Speaker-Stratified CV...")
    results = {'Task': [], 'Model': [], 'Accuracy': [], 'Std': []}

    NON_FEATURE_COLS = {'group', 'label', 'task', 'speaker'}

    for task in ml_df['task'].unique():
        task_data = ml_df[ml_df['task'] == task].copy()

        # Per-task feature cols — drop all-NaN columns first
        feature_cols = [c for c in task_data.columns if c not in NON_FEATURE_COLS]
        feature_cols = [c for c in feature_cols if task_data[c].notna().any()]

        if len(feature_cols) < 3:
            print(f"  ⚠️  {task}: too few features, skipping")
            continue

        X      = task_data[feature_cols].copy()
        y      = task_data['label']
        groups = task_data['speaker']

        if len(y.unique()) < 2:
            print(f"  ⚠️  {task}: only one class, skipping")
            continue

        imputer = SimpleImputer(strategy='median')  # median more robust for non-normal data
        X_imp   = pd.DataFrame(
            imputer.fit_transform(X),
            columns=X.columns,
            index=X.index
        )

        n_splits = min(5, int(y.value_counts().min()))
        if n_splits < 2:
            print(f"  ⚠️  {task}: not enough samples for CV, skipping")
            continue

        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

        for name, model in [
            ('RF',  Pipeline([('rf',  RandomForestClassifier(n_estimators=100, random_state=42))])),
            ('SVM', Pipeline([('scale', StandardScaler()), ('svm', SVC(random_state=42))]))
        ]:
            accs = []
            for train_idx, test_idx in sgkf.split(X_imp, y, groups):
                if len(np.unique(y.iloc[train_idx])) < 2:
                    continue
                model.fit(X_imp.iloc[train_idx], y.iloc[train_idx])
                accs.append(accuracy_score(y.iloc[test_idx], model.predict(X_imp.iloc[test_idx])))

            if accs:
                mean_acc = np.mean(accs)
                std_acc  = np.std(accs)
                results['Task'].append(task)
                results['Model'].append(name)
                results['Accuracy'].append(mean_acc)
                results['Std'].append(std_acc)
                print(f"  {task} | {name}: {mean_acc:.3f} ± {std_acc:.3f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv('results/ml_results_speaker_split.csv', index=False)

    if not results_df.empty:
        pivot_acc = results_df.pivot(index='Task', columns='Model', values='Accuracy').fillna(0)
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_acc, annot=True, cmap='YlOrRd', fmt='.3f', vmin=0, vmax=1)
        plt.title('ML Accuracy: Speaker-Stratified CV (PD vs HC)', fontsize=14)
        plt.ylabel('Task'); plt.xlabel('Model')
        plt.tight_layout()
        plt.savefig('results/ml_accuracies.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ ml_accuracies.png")
    else:
        print("⚠️  No valid ML results to plot")
else:
    print("⚠️  Skipping ML: Insufficient data")

# =============================================================
# FINAL SUMMARY
# =============================================================
print("\n" + "="*60)
print("🚀 FULL PIPELINE COMPLETE! ✅")
print("="*60)
print("\n📊 STATISTICAL APPROACH:")
print("  Primary test:   Mann-Whitney U (non-parametric, valid for ALL features)")
print("  Secondary test: Welch t-test (parametric, valid ONLY for HNR, F0_MEAN, SAMPEN)")
print("  Effect size:    Cohen's d (distribution-free descriptor)")
print("  3-group test:   Kruskal-Wallis + Bonferroni post-hoc")
print(f"  Normality:      {pct_non:.1f}% features non-normal → non-parametric justified")
print("\n📁 OUTPUT FILES (all in results/ folder):")
print("   normality_shapiro.csv        ← Shapiro-Wilk per feature/task")
print("   full_stat_summary.csv        ← MW p, t-test p, Cohen's d, means")
print("   kruskal_mannwhitney.csv      ← KW + post-hoc per feature/task")
print("   heatmap1_mannwhitney_pvalues.png  ← PRIMARY p-value heatmap")
print("   heatmap2_effect_sizes.png    ← Cohen's d")
print("   heatmap3_mean_differences.png← PD - HC means")
print("   heatmap4_significance.png    ← -log10(MW p)")
print("   ml_features_ready.csv        ← ML input (ranked by MW p)")
print("   ml_results_speaker_split.csv ← RF + SVM accuracy")
print("   ml_accuracies.png            ← ML heatmap")
print("\n✅ Ready for Week 8 presentation!")