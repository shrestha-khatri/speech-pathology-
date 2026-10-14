"""
Elderly Healthy Control Numerical Analysis Script

This script performs statistical analysis on the extracted speech features
for the Elderly Healthy Control dataset.

The analysis includes:
• Descriptive statistics by task
• Pairwise Mann–Whitney U tests between tasks
• Kruskal–Wallis omnibus tests across tasks
• Effect size calculation (Cohen’s d)
• Automatic export of results to a formatted Excel workbook

Input:
elderlyHC_paper_features.csv

Output:
ElderlyHC_Analysis_<timestamp>.xlsx containing:
    Summary statistics
    Pairwise comparisons
    Kruskal–Wallis results
    Interpretation table
    Raw dataset
"""

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, kruskal
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment
from datetime import datetime


print("Elderly HC Numerical Analysis")

# Load extracted feature dataset
df = pd.read_csv(r"C:\Users\DELL\Codes\DH604\elderly analysis\elderlyHC_paper_features.csv")
print(f"Loaded {len(df)} records")
print(f"Tasks present: {df['task'].value_counts().to_dict()}")

# Features selected for statistical analysis
features = ['spir', 'ppr', 'hnr', 'jitter_local', 'shimmer_local']


# Compute descriptive statistics grouped by speech task
print("Computing summary statistics")
summary_stats = df.groupby('task')[features].agg(
    ['count', 'mean', 'std', 'median', 'min', 'max']
).round(3)


# Pairwise Mann–Whitney U tests between task types
print("Running pairwise tests")
pairwise_results = []
tasks_list = [t for t in df['task'].unique() if t != 'other']

for feat in features:
    for i, task1 in enumerate(tasks_list):
        for task2 in tasks_list[i+1:]:

            data1 = df[df['task'] == task1][feat].dropna()
            data2 = df[df['task'] == task2][feat].dropna()

            if len(data1) >= 3 and len(data2) >= 3:
                stat, p = mannwhitneyu(data1, data2)

                # Cohen's d effect size calculation
                n1, n2 = len(data1), len(data2)
                pooled_std = np.sqrt(((n1-1)*data1.std()**2 + (n2-1)*data2.std()**2) / (n1+n2-2))
                cohens_d = (data1.mean() - data2.mean()) / pooled_std if pooled_std > 0 else 0

                pairwise_results.append({
                    'Feature': feat.upper(),
                    'Comparison': f'{task1} vs {task2}',
                    'G1_Mean': round(data1.mean(), 3),
                    'G2_Mean': round(data2.mean(), 3),
                    'n1': n1,
                    'n2': n2,
                    'Diff': round(data1.mean() - data2.mean(), 3),
                    'U': round(stat, 2),
                    'p': round(p, 4),
                    'Sig': 'YES' if p < 0.05 else 'NO',
                    'd': round(cohens_d, 3)
                })

pairwise_df = pd.DataFrame(pairwise_results)


# Kruskal–Wallis omnibus tests across all tasks
print("Running Kruskal–Wallis tests")
omnibus_results = []

for feat in features:
    data_by_task = [df[df['task'] == t][feat].dropna().values for t in tasks_list]

    valid_tasks = []
    valid_data = []
    for i, d in enumerate(data_by_task):
        if len(d) >= 2 and np.std(d) > 0:
            valid_tasks.append(tasks_list[i])
            valid_data.append(d)

    if len(valid_data) >= 2:
        try:
            h_stat, p = kruskal(*valid_data)
            omnibus_results.append({
                'Feature': feat.upper(),
                'Tasks': ', '.join(valid_tasks),
                'H': round(h_stat, 3),
                'p': round(p, 4),
                'Sig': 'YES' if p < 0.05 else 'NO'
            })
        except:
            omnibus_results.append({
                'Feature': feat.upper(),
                'Tasks': ', '.join(valid_tasks),
                'H': 'N/A',
                'p': 1.000,
                'Sig': 'NO (zeros)'
            })
    else:
        omnibus_results.append({
            'Feature': feat.upper(),
            'Tasks': 'No variance',
            'H': 'N/A',
            'p': 1.000,
            'Sig': 'NO'
        })

omnibus_df = pd.DataFrame(omnibus_results)


# Interpretation sheet summarising mean values per task
interp_df = pd.DataFrame({
    'Feature': [f.upper() for f in features],
    'Definition': ['Pauses/min', 'Pause Ratio', 'HNR', 'Jitter Local', 'Shimmer Local'],
    'Simple': [round(df[df['task']=='simple'][f].mean(), 3) for f in features],
    'Reading': [round(df[df['task']=='reading'][f].mean(), 3) for f in features],
    'Diado': [round(df[df['task']=='diadochokinetic'][f].mean(), 3) for f in features],
    'Expected_PD_Pattern': ['↑', '↑', '↓', '↑', '↑']
})


# Save all results into Excel workbook
excel_file = f'ElderlyHC_Analysis_{datetime.now().strftime("%Y%m%d_%H%M")}.xlsx'
with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
    summary_stats.to_excel(writer, 'Summary')
    pairwise_df.to_excel(writer, 'Pairwise', index=False)
    omnibus_df.to_excel(writer, 'Kruskal', index=False)
    interp_df.to_excel(writer, 'Interpretation', index=False)
    df.to_excel(writer, 'Raw')

print(f"Excel file saved: {excel_file}")


# Quick console preview of key results
print("\nMean feature values by task")
print(df.groupby('task')[features].mean().round(3))

print("\nSignificant pairwise differences")
print(pairwise_df[pairwise_df['Sig'] == 'YES'])
