"""
Task-Based Phonatory Analysis Pipeline

This script performs statistical analysis of phonatory features across
multiple speech tasks and three participant groups:
youngHC, elderlyHC, parkinsonPD.

Pipeline steps
1. Load and clean dataset
2. Compute descriptive statistics by task and group
3. Run ANOVA per task
4. Perform Tukey HSD post-hoc tests
5. Create summary tables for reporting
6. Generate publication-quality visualizations
7. Compute Cohen’s d effect sizes

Input file
phonatory_task_grouped.csv

Outputs
task_stat_summary.csv
task_effect_sizes.csv
task_boxplots.png
task_violins.png
task_means.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.anova import anova_lm
import warnings
warnings.filterwarnings('ignore')

print("Complete task based phonatory analysis pipeline")


# Load dataset containing phonatory features extracted earlier
print("Loading and cleaning data")
df = pd.read_csv(r"phonatory_task_grouped.csv")

# Columns that must be numeric for statistical analysis
numeric_cols = [
    'hnr','jitter_local','jitter_rap','jitter_ppq5',
    'shimmer_local','shimmer_apq3','shimmer_apq5',
    'f0_mean','f0_std'
]

# Convert columns safely to numeric values
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print("Original shape:", df.shape)
print("Groups:", df['group'].value_counts().to_dict())
print("Tasks:", sorted(df['task'].unique()))

# Show how many recordings exist per task and group
print("Rows by task and group")
print(df.groupby(['task','group']).size().unstack(fill_value=0))


# Keep only tasks where all three groups exist
task_counts = df.groupby('task')['group'].nunique()
valid_tasks = task_counts[task_counts == 3].index

# Remove rows missing key biomarkers required for analysis
df_analysis = df[
    df['task'].isin(valid_tasks)
].dropna(subset=['hnr','jitter_local','shimmer_local','f0_mean'])

print("Analysis dataset:", df_analysis.shape)


# Core phonatory biomarkers used in Parkinson’s research
print("Descriptive statistics by task and group")
key_features = ['hnr','jitter_local','shimmer_local','f0_mean']

summary_stats = df_analysis.groupby(
    ['task','group']
)[key_features].agg(['mean','std','count']).round(4)

print(summary_stats)


# Run one-way ANOVA per task comparing the three groups
print("ANOVA per task")
anova_results = {}

for task in valid_tasks:
    df_task = df_analysis[df_analysis['task'] == task]
    for feature in key_features:
        model = ols(f'{feature} ~ C(group)', data=df_task).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        anova_results[f"{task}_{feature}"] = anova_table
        print(task, feature)
        print(anova_table.round(4))


# Tukey HSD post-hoc pairwise comparisons
print("Tukey HSD post hoc tests")
tukey_results = {}

for task in valid_tasks:
    df_task = df_analysis[df_analysis['task'] == task]
    print("Task:", task)
    for feature in key_features:
        tukey = pairwise_tukeyhsd(
            endog=df_task[feature],
            groups=df_task['group'],
            alpha=0.05
        )
        print(feature)
        print(tukey)
        tukey_results[f"{task}_{feature}"] = tukey


# Build summary table formatted for paper reporting
print("Creating statistical summary table")
stat_summary = []

for task in valid_tasks:
    df_task = df_analysis[df_analysis['task'] == task]
    for feature in key_features:
        group_means = df_task.groupby('group')[feature].mean()
        group_sds = df_task.groupby('group')[feature].std()
        anova_p = anova_results[f"{task}_{feature}"].loc['C(group)','PR(>F)']

        stat_summary.append({
            'Task': task,
            'Feature': feature.upper(),
            'YoungHC': f"{group_means.get('youngHC','NA'):.2f}±{group_sds.get('youngHC','NA'):.2f}",
            'ElderlyHC': f"{group_means.get('elderlyHC','NA'):.2f}±{group_sds.get('elderlyHC','NA'):.2f}",
            'ParkinsonPD': f"{group_means.get('parkinsonPD','NA'):.2f}±{group_sds.get('parkinsonPD','NA'):.2f}",
            'ANOVA p': f"{anova_p:.4f}",
            'Sig Pairs': 'See Tukey'
        })

summary_df = pd.DataFrame(stat_summary)
summary_df.to_csv('task_stat_summary.csv', index=False)
print(summary_df)


# Generate publication-quality plots
print("Generating visualizations")
plt.style.use('default')
sns.set_palette("husl")


# Boxplots showing distribution by task and group
fig, axes = plt.subplots(2,2, figsize=(16,12))
axes = axes.ravel()

for idx, feature in enumerate(key_features):
    sns.boxplot(data=df_analysis, x='task', y=feature, hue='group', ax=axes[idx])
    axes[idx].set_title(feature.upper())
    axes[idx].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('task_boxplots.png', dpi=300)
plt.show()


# Violin plots for distribution visualization
fig, axes = plt.subplots(2,2, figsize=(16,10))
axes = axes.ravel()

for idx, feature in enumerate(key_features):
    sns.violinplot(data=df_analysis, x='task', y=feature, hue='group', split=True, ax=axes[idx])
    axes[idx].set_title(feature.upper())
    axes[idx].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('task_violins.png', dpi=300)
plt.show()


# Mean ± SEM bar plots
fig, axes = plt.subplots(2,2, figsize=(16,12))
tasks_list = sorted(valid_tasks)

for idx, feature in enumerate(key_features):
    ax = axes[idx//2, idx%2]
    task_means = df_analysis.groupby(['task','group'])[feature].mean().unstack()
    task_sems = df_analysis.groupby(['task','group'])[feature].sem().unstack()

    x = np.arange(len(tasks_list))
    width = 0.25
    groups = ['youngHC','elderlyHC','parkinsonPD']

    for i, group in enumerate(groups):
        means = task_means[group].reindex(tasks_list)
        sems = task_sems[group].reindex(tasks_list)
        ax.bar(x + i*width, means, width, yerr=sems, capsize=5, label=group)

    ax.set_title(feature.upper())
    ax.set_xticks(x + width)
    ax.set_xticklabels(tasks_list, rotation=45)
    ax.legend()

plt.tight_layout()
plt.savefig('task_means.png', dpi=300)
plt.show()


# Compute Cohen’s d effect sizes between groups
print("Computing Cohen d effect sizes")

def cohens_d(group1, group2):
    return (group1.mean() - group2.mean()) / np.sqrt((group1.var() + group2.var()) / 2)

young_data = df_analysis[df_analysis['group'] == 'youngHC']
elderly_data = df_analysis[df_analysis['group'] == 'elderlyHC']
pd_data = df_analysis[df_analysis['group'] == 'parkinsonPD']

effect_sizes = pd.DataFrame(
    index=key_features,
    columns=['Young-Elderly','Elderly-PD','PD-Young']
)

for feature in key_features:
    effect_sizes.loc[feature,'Young-Elderly'] = f"{cohens_d(young_data[feature], elderly_data[feature]):.3f}"
    effect_sizes.loc[feature,'Elderly-PD'] = f"{cohens_d(elderly_data[feature], pd_data[feature]):.3f}"
    effect_sizes.loc[feature,'PD-Young'] = f"{cohens_d(pd_data[feature], young_data[feature]):.3f}"

effect_sizes.to_csv('task_effect_sizes.csv')
print(effect_sizes)

print("Analysis complete")
