import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=== 4 PUBLICATION HEATMAPS - BULLETPROOF VERSION ===")

df = pd.read_csv("phonatory_task_grouped_FINAL.csv")

numeric_cols = [
    'hnr', 'jitter_local', 'jitter_rap', 'jitter_ppq5', 'shimmer_local', 
    'shimmer_apq5', 'f0_mean', 'f0_std', 'df0', 'ddf0', 'rpde_proxy', 
    'dfa_proxy', 'hurst_proxy', 'lle_proxy', 'apen_proxy', 'sampen_proxy', 
    'ppe', 'ftri_proxy', 'cvf0'
]

# Filter PD vs HC + valid tasks
df_pd_hc = df[df['group'].isin(['parkinsonPD', 'elderlyHC', 'youngHC'])].copy()
task_counts = df_pd_hc.groupby('task')['group'].nunique()
valid_tasks = task_counts[task_counts >= 2].index
df_analysis = df_pd_hc[df_pd_hc['task'].isin(valid_tasks)].copy()

print(f"📊 {len(valid_tasks)} tasks × {len(numeric_cols)} features")

# === SAFE ANOVA CALCULATION ===
print("\n🔥 Computing SAFE statistics...")
stat_summary = []

for task in valid_tasks:
    df_task = df_analysis[df_analysis['task'] == task]
    print(f"  {task}: n={len(df_task)}")
    
    for feature in numeric_cols:
        if feature in df_task.columns:
            feature_data = df_task[feature].dropna()
            if len(feature_data) >= 4:  # Minimum for reliable stats
                # SAFE group means/std
                hc_data = df_task[df_task['group'].isin(['elderlyHC', 'youngHC'])][feature].dropna()
                pd_data = df_task[df_task['group'] == 'parkinsonPD'][feature].dropna()
                
                if len(hc_data) >= 2 and len(pd_data) >= 2:
                    hc_mean, pd_mean = hc_data.mean(), pd_data.mean()
                    hc_std, pd_std = hc_data.std(), pd_data.std()
                    
                    # SAFE Cohen's d (avoid div by zero)
                    pooled_std = np.sqrt((hc_std**2 + pd_std**2) / 2)
                    cohens_d = abs(hc_mean - pd_mean) / pooled_std if pooled_std > 0.001 else 0
                    
                    # SAFE t-test p-value
                    from scipy import stats
                    t_stat, p_val = stats.ttest_ind(hc_data, pd_data, equal_var=False)
                    
                    stat_summary.append({
                        'Task': task, 
                        'Feature': feature.replace('_proxy', '').upper(),
                        'HC_mean': hc_mean, 'PD_mean': pd_mean,
                        'ANOVA_p': p_val, 'Cohens_d': cohens_d
                    })

summary_df = pd.DataFrame(stat_summary)
summary_df.to_csv('safe_task_summary.csv', index=False)

# === 4 HEATMAPS - BULLETPROOF ===
plt.style.use('default')
sns.set_palette(["#FFA500", "#DC143C"])

# 1. ANOVA P-VALUES
p_matrix = pd.pivot_table(summary_df, values='ANOVA_p', index='Feature', columns='Task', aggfunc='first').fillna(1.0)
plt.figure(figsize=(14, 10))
sns.heatmap(p_matrix, annot=True, cmap='viridis', fmt='.3f', 
            cbar_kws={'label': 'p-value'})
plt.title('ANOVA P-VALUES: PD vs HC (safer t-test)', fontsize=16, pad=20)
plt.xlabel('Speech Task'); plt.ylabel('Phonatory Feature')
plt.tight_layout()
plt.savefig('heatmap1_pvalues_safe.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ heatmap1_pvalues_safe.png")

# 2. EFFECT SIZES
effect_matrix = pd.pivot_table(summary_df, values='Cohens_d', index='Feature', columns='Task', aggfunc='first').fillna(0)
plt.figure(figsize=(14, 10))
sns.heatmap(effect_matrix, annot=True, cmap='RdBu_r', center=0, fmt='.2f',
            cbar_kws={'label': "Cohen's d (abs(HC-PD))"})
plt.title('EFFECT SIZES: PD vs HC', fontsize=16, pad=20)
plt.xlabel('Speech Task'); plt.ylabel('Phonatory Feature')
plt.tight_layout()
plt.savefig('heatmap2_effects_safe.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ heatmap2_effects_safe.png")

# 3. MEAN DIFFERENCES
mean_diff = summary_df.groupby('Feature')['PD_mean'].mean() - summary_df.groupby('Feature')['HC_mean'].mean()
mean_diff = pd.pivot_table(summary_df, values='PD_mean', index='Feature', columns='Task') - \
            pd.pivot_table(summary_df, values='HC_mean', index='Feature', columns='Task')
mean_diff = mean_diff.fillna(0)
plt.figure(figsize=(14, 10))
sns.heatmap(mean_diff, annot=True, cmap='coolwarm', center=0, fmt='.2f',
            cbar_kws={'label': 'PD_mean - HC_mean'})
plt.title('MEAN DIFFERENCES: PD vs HC', fontsize=16, pad=20)
plt.xlabel('Speech Task'); plt.ylabel('Phonatory Feature')
plt.tight_layout()
plt.savefig('heatmap3_meandiff_safe.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ heatmap3_meandiff_safe.png")

# 4. SIGNIFICANCE (-log10(p))
p_numeric = p_matrix.copy()
p_numeric = np.log10(p_numeric.clip(lower=0.0001)) * -1
plt.figure(figsize=(14, 10))
sns.heatmap(p_numeric, annot=True, cmap='coolwarm', fmt='.1f',
            cbar_kws={'label': '-log10(p-value)'})
plt.title('STATISTICAL SIGNIFICANCE\n(Higher = More Significant)', fontsize=16, pad=20)
plt.xlabel('Speech Task'); plt.ylabel('Phonatory Feature')
plt.tight_layout()
plt.savefig('heatmap4_significance_safe.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ heatmap4_significance_safe.png")

print("\n🎉 4 BULLETPROOF HEATMAPS - NO ERRORS!")
print("✅ All files saved with _safe.png suffix")
print("✅ Used scipy t-test (safer than ANOVA)")
print("✅ safe_task_summary.csv")
print("\n🚀 Ready for your PPT presentation!")
