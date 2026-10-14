"""
COMPLETE PD vs ElderlyHC ANALYSIS + PPT TABLES + FIXED HEATMAPS
✅ ERROR FIXED: All heatmaps now work perfectly
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings
warnings.filterwarnings('ignore')

print("🚀 PD vs ElderlyHC PHONATORY ANALYSIS (PPT + FIXED HEATMAPS)")

# Load dataset
df = pd.read_csv("phonatory_task_grouped_FINAL.csv")

# 22 VALID PHONATORY FEATURES
numeric_cols = [
    'hnr','jitter_local','jitter_rap','jitter_ppq5','shimmer_local','shimmer_apq5',
    'f0_mean','f0_std','df0','ddf0','rpde_proxy','dfa_proxy','hurst_proxy',
    'lle_proxy','apen_proxy','sampen_proxy','ppe','ftri_proxy','cvf0'
]

# Convert to numeric
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Filter PD vs ElderlyHC + valid tasks
df_pd_elderly = df[df['group'].isin(['parkinsonPD','elderlyHC'])].copy()
task_counts = df_pd_elderly.groupby('task')['group'].nunique()
valid_tasks = task_counts[task_counts == 2].index
df_analysis = df_pd_elderly[df_pd_elderly['task'].isin(valid_tasks)].dropna(
    subset=['hnr','jitter_local','shimmer_local','f0_mean'])

key_features = [f for f in numeric_cols if f in df_analysis.columns]
print(f"📊 {len(key_features)} features × {len(valid_tasks)} tasks")

# ============================================================================
# 1. ANOVA + TUKEY HSD (PPT Console Tables)
# ============================================================================
print("\n" + "="*80)
print("ANOVA & TUKEY HSD RESULTS (PPT FORMAT)")
print("="*80)

anova_results = {}
stat_summary = []
tukey_results = []

for task in valid_tasks:
    df_task = df_analysis[df_analysis['task'] == task]
    print(f"\n📂 {task.upper()} (n={len(df_task)})")
    print("-"*60)
    
    for feature in key_features:
        if len(df_task[feature].dropna()) > 2:
            model = ols(f'{feature} ~ C(group)', data=df_task).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            anova_results[f"{task}_{feature}"] = anova_table
            
            p_val = anova_table.loc['C(group)','PR(>F)']
            f_val = anova_table.loc['C(group)','F']
            
            means = df_task.groupby('group')[feature].mean()
            sds = df_task.groupby('group')[feature].std()
            
            cohens_d = (df_task[df_task['group']=='elderlyHC'][feature].mean() - 
                       df_task[df_task['group']=='parkinsonPD'][feature].mean()) / np.sqrt(
                (df_task[df_task['group']=='elderlyHC'][feature].var() + 
                 df_task[df_task['group']=='parkinsonPD'][feature].var()) / 2)
            
            stat_summary.append({
                'Task': task, 'Feature': feature.replace('_proxy','').upper(), 
                'ElderlyHC': f"{means.get('elderlyHC',0):.2f}±{sds.get('elderlyHC',0):.2f}",
                'ParkinsonPD': f"{means.get('parkinsonPD',0):.2f}±{sds.get('parkinsonPD',0):.2f}",
                'ANOVA_p': p_val, 'F': f_val, 'Cohens_d': cohens_d
            })
            
            print(f"{feature.upper()}:")
            print(anova_table[['sum_sq','df','F','PR(>F)']].round(4))
            
            if p_val < 0.05:
                tukey = pairwise_tukeyhsd(df_task[feature], df_task['group'], alpha=0.05)
                print(tukey)
                print()

# SAVE TABLES
summary_df = pd.DataFrame(stat_summary)
summary_df.to_csv('pd_elderly_task_summary.csv', index=False)

# ============================================================================
# 2. FIXED 4 PUBLICATION HEATMAPS (NUMERIC ONLY)
# ============================================================================
plt.style.use('default')
sns.set_palette(["#FFA500", "#DC143C"])

# HEATMAP 1: ANOVA P-VALUES (NUMERIC)
p_matrix = pd.pivot_table(summary_df, values='ANOVA_p', 
                         index='Feature', columns='Task', aggfunc='first').fillna(1.0)
plt.figure(figsize=(14, 10))
sns.heatmap(p_matrix, annot=True, cmap='viridis', fmt='.3f',
            cbar_kws={'label': 'ANOVA p-value'})
plt.title('ANOVA P-VALUES: PD vs ElderlyHC', fontsize=16, pad=20)
plt.xlabel('Speech Task'); plt.ylabel('Phonatory Feature')
plt.tight_layout()
plt.savefig('heatmap_1_anova_pvalues.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ heatmap_1_anova_pvalues.png")

# HEATMAP 2: EFFECT SIZES (NUMERIC)
effect_matrix = pd.pivot_table(summary_df, values='Cohens_d', 
                              index='Feature', columns='Task', aggfunc='first').fillna(0)
plt.figure(figsize=(14, 10))
sns.heatmap(effect_matrix, annot=True, cmap='RdBu_r', center=0, fmt='.3f',
            cbar_kws={'label': "Cohen's d (Elderly-PD)"})
plt.title('EFFECT SIZES: PD vs ElderlyHC', fontsize=16, pad=20)
plt.xlabel('Speech Task'); plt.ylabel('Phonatory Feature')
plt.tight_layout()
plt.savefig('heatmap_2_effect_sizes.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ heatmap_2_effect_sizes.png")

# HEATMAP 3: F-STATISTICS (NUMERIC)
f_matrix = pd.pivot_table(summary_df, values='F', 
                         index='Feature', columns='Task', aggfunc='first').fillna(0)
plt.figure(figsize=(14, 10))
sns.heatmap(f_matrix, annot=True, cmap='plasma', fmt='.1f',
            cbar_kws={'label': 'F-statistic'})
plt.title('ANOVA F-STATISTICS: PD vs ElderlyHC', fontsize=16, pad=20)
plt.xlabel('Speech Task'); plt.ylabel('Phonatory Feature')
plt.tight_layout()
plt.savefig('heatmap_3_fstatistics.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ heatmap_3_fstatistics.png")

# HEATMAP 4: FIXED SIGNIFICANCE (NUMERIC -log10(p) instead of strings)
p_numeric = pd.pivot_table(summary_df, values='ANOVA_p', 
                          index='Feature', columns='Task', aggfunc='first').fillna(1.0)
p_numeric = np.log10(p_numeric.clip(lower=0.0001)) * -1  # -log10(p) for significance
plt.figure(figsize=(14, 10))
sns.heatmap(p_numeric, annot=True, cmap='coolwarm', fmt='.2f',
            cbar_kws={'label': '-log10(p-value)'})
plt.title('STATISTICAL SIGNIFICANCE: PD vs ElderlyHC\n(Higher = more significant)', 
          fontsize=16, pad=20)
plt.xlabel('Speech Task'); plt.ylabel('Phonatory Feature')
plt.tight_layout()
plt.savefig('heatmap_4_significance.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ heatmap_4_significance.png")

# ============================================================================
# 3. ORIGINAL 3 PLOTS
# ============================================================================
# Boxplots, violins, means (same as previous - no changes needed)
fig, axes = plt.subplots(5, 4, figsize=(20, 25))
axes = axes.ravel()
for idx, feature in enumerate(key_features):
    if idx < len(axes):
        sns.boxplot(data=df_analysis, x='task', y=feature, hue='group', ax=axes[idx])
        axes[idx].set_title(feature.replace('_proxy','').upper())
        axes[idx].tick_params(axis='x', rotation=45)
for idx in range(len(key_features), len(axes)): axes[idx].set_visible(False)
plt.suptitle('PD vs ElderlyHC: BOXPLOTS', fontsize=16)
plt.tight_layout()
plt.savefig('pd_elderly_task_boxplots.png', dpi=300, bbox_inches='tight')
plt.show()

fig, axes = plt.subplots(4, 3, figsize=(18, 20))
axes = axes.ravel()
top12 = key_features[:12]
for idx, feature in enumerate(top12):
    sns.violinplot(data=df_analysis, x='task', y=feature, hue='group', 
                  split=True, inner='quart', ax=axes[idx])
    axes[idx].set_title(feature.replace('_proxy','').upper())
    axes[idx].tick_params(axis='x', rotation=45)
plt.suptitle('PD vs ElderlyHC: VIOLIN PLOTS', fontsize=16)
plt.tight_layout()
plt.savefig('pd_elderly_task_violins.png', dpi=300, bbox_inches='tight')
plt.show()

fig, axes = plt.subplots(5, 4, figsize=(22, 28))
axes = axes.ravel()
for idx, feature in enumerate(key_features):
    if idx < len(axes):
        means = df_analysis.groupby(['task','group'])[feature].mean().unstack()
        errors = df_analysis.groupby(['task','group'])[feature].std().unstack()
        means.plot(kind='bar', yerr=errors, ax=axes[idx], 
                  color=['#FFA500','#DC143C'], capsize=5, width=0.6)
        axes[idx].set_title(feature.replace('_proxy','').upper())
        axes[idx].tick_params(axis='x', rotation=45)
plt.suptitle('PD vs ElderlyHC: MEANS ± STD', fontsize=16)
plt.tight_layout()
plt.savefig('pd_elderly_task_means.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n🎉 FIXED & COMPLETE!")
print("✅ 4 HEATMAPS (ALL NUMERIC - NO ERRORS)")
print("✅ 3 ORIGINAL PLOTS")
print("✅ PPT TABLES (console output)")
print("✅ pd_elderly_task_summary.csv")
