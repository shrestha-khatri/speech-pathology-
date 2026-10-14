"""
PD vs ElderlyHC Task-Based Phonatory Analysis Script

This script performs statistical analysis of phonatory voice features
comparing Parkinson’s Disease participants against age-matched
Elderly Healthy Controls across ALL available speech tasks.

Pipeline overview
1. Load and clean phonatory_task_grouped.csv
2. Filter dataset to keep only ParkinsonPD and ElderlyHC groups
3. Keep only tasks where BOTH groups are present
4. Compute descriptive statistics
5. Run per-task ANOVA
6. Run Tukey HSD post-hoc comparisons
7. Generate publication-quality plots
8. Compute Cohen’s d effect sizes

Outputs produced by this script
pd_elderly_task_summary.csv
pd_elderly_effect_sizes.csv
pd_elderly_task_boxplots.png
pd_elderly_task_violins.png
pd_elderly_task_means.png
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


print("PD vs ElderlyHC phonatory analysis")


# Load dataset containing phonatory features extracted earlier
df = pd.read_csv(r"D:\Download\Codes\DH604\vowel based week4\phonatory_task_grouped.csv")


# Columns that must be numeric for statistical analysis
numeric_cols = [
    'hnr','jitter_local','jitter_rap','jitter_ppq5',
    'shimmer_local','shimmer_apq3','shimmer_apq5',
    'f0_mean','f0_std'
]

# Convert columns safely to numeric values
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')


# Filter dataset to keep only Parkinson’s Disease and Elderly controls
df_pd_elderly = df[df['group'].isin(['parkinsonPD','elderlyHC'])].copy()

print("Dataset shape:", df_pd_elderly.shape)
print(df_pd_elderly['group'].value_counts())

# Show how many recordings exist per task and group
print(df_pd_elderly.groupby(['task','group']).size().unstack(fill_value=0))


# Keep only tasks where BOTH groups exist
task_counts = df_pd_elderly.groupby('task')['group'].nunique()
valid_tasks = task_counts[task_counts == 2].index

# Drop rows missing key features required for statistical testing
df_analysis = df_pd_elderly[
    df_pd_elderly['task'].isin(valid_tasks)
].dropna(subset=['hnr','jitter_local','shimmer_local','f0_mean'])

print("Analysis dataset:", df_analysis.shape)


# Core phonatory biomarkers used in Parkinson’s literature
key_features = ['hnr','jitter_local','shimmer_local','f0_mean']


# Compute descriptive statistics per task and group
summary_stats = df_analysis.groupby(['task','group'])[key_features].agg(
    ['mean','std','count']
).round(4)

print(summary_stats)


# Run one-way ANOVA per task comparing PD vs ElderlyHC
anova_results = {}

for task in valid_tasks:
    df_task = df_analysis[df_analysis['task'] == task]
    for feature in key_features:
        model = ols(f'{feature} ~ C(group)', data=df_task).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        anova_results[f"{task}_{feature}"] = anova_table
        print(task, feature)
        print(anova_table.round(4))


# Tukey HSD post-hoc comparisons per task
print("Tukey HSD")
for task in valid_tasks:
    df_task = df_analysis[df_analysis['task'] == task]
    print("Task:", task)
    for feature in key_features:
        tukey = pairwise_tukeyhsd(df_task[feature], df_task['group'], alpha=0.05)
        print(feature)
        print(tukey)


# Build summary table formatted for paper reporting
stat_summary = []

for task in valid_tasks:
    df_task = df_analysis[df_analysis['task'] == task]
    for feature in key_features:
        means = df_task.groupby('group')[feature].mean()
        sds = df_task.groupby('group')[feature].std()
        p_val = anova_results[f"{task}_{feature}"].loc['C(group)','PR(>F)']

        stat_summary.append({
            'Task': task,
            'Feature': feature.upper(),
            'ElderlyHC': f"{means.get('elderlyHC','NA'):.2f}±{sds.get('elderlyHC','NA'):.2f}",
            'ParkinsonPD': f"{means.get('parkinsonPD','NA'):.2f}±{sds.get('parkinsonPD','NA'):.2f}",
            'ANOVA p': f"{p_val:.4f}"
        })

summary_df = pd.DataFrame(stat_summary)
summary_df.to_csv('pd_elderly_task_summary.csv', index=False)


# Set plotting style for publication-ready figures
plt.style.use('default')
sns.set_palette(["orange","red"])


# Boxplots showing feature distribution across tasks
fig, axes = plt.subplots(2,2, figsize=(16,12))
axes = axes.ravel()

for idx, feature in enumerate(key_features):
    sns.boxplot(data=df_analysis, x='task', y=feature, hue='group', ax=axes[idx])
    axes[idx].set_title(feature.upper())
    axes[idx].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('pd_elderly_task_boxplots.png', dpi=300)
plt.show()


# Violin plots for distribution visualization
fig, axes = plt.subplots(2,2, figsize=(16,10))
axes = axes.ravel()

for idx, feature in enumerate(key_features):
    sns.violinplot(data=df_analysis, x='task', y=feature, hue='group', split=True, ax=axes[idx])
    axes[idx].set_title(feature.upper())
    axes[idx].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('pd_elderly_task_violins.png', dpi=300)
plt.show()


# Function to compute Cohen’s d effect size
def cohens_d(group1, group2):
    return (group1.mean() - group2.mean()) / np.sqrt((group1.var() + group2.var()) / 2)

elderly_data = df_analysis[df_analysis['group'] == 'elderlyHC']
pd_data = df_analysis[df_analysis['group'] == 'parkinsonPD']

effect_sizes = pd.DataFrame(index=key_features, columns=['Elderly-PD','PD-Elderly'])

for feature in key_features:
    effect_sizes.loc[feature,'Elderly-PD'] = f"{cohens_d(elderly_data[feature], pd_data[feature]):.3f}"
    effect_sizes.loc[feature,'PD-Elderly'] = f"{cohens_d(pd_data[feature], elderly_data[feature]):.3f}"

effect_sizes.to_csv('pd_elderly_effect_sizes.csv')

print("Analysis complete")
