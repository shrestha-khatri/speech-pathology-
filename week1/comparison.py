"""
Speech Feature Group Comparison Script

This script compares speech features across three participant groups:
1) elderlyHC   (Elderly Healthy Controls)
2) parkinsonPD (Parkinson’s Disease patients)
3) youngHC     (Young Healthy Controls)

The script performs:
• Data loading from CSV feature files
• Descriptive statistics (mean, std, range, N)
• Kruskal–Wallis statistical testing
• Boxplot visualisation of SPIR and PPR

Features analysed:
duration, spir (speech rate), ppr (pauses per reading), n_pauses

Outputs:
spir_ppr_comparison.csv  → summary statistics
spir_ppr_comparison.png  → boxplots
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


# File paths for the three datasets
files = {
    'elderlyHC': r'C:\Users\DELL\Codes\DH604\elderly analysis\elderlyHC_paper_features.csv',
    'parkinsonPD': r'C:\Users\DELL\Codes\DH604\pd analysis\parkinsonPD_paper_features.csv',
    'youngHC': r'C:\Users\DELL\Codes\DH604\young healthy analysis\youngHC_paper_features.csv'
}

# Load CSV files into pandas DataFrames
datasets = {name: pd.read_csv(fname, encoding='latin1') for name, fname in files.items()}

# Features chosen because they contain meaningful variation
meaningful_features = ['duration', 'spir', 'ppr', 'n_pauses']

print("Dataset sizes:", {k: len(v) for k, v in datasets.items()})


# Compute descriptive statistics for each group
summary = []
for name, df in datasets.items():
    row = {'Group': name}

    for feat in meaningful_features:
        # Convert column to numeric and remove invalid entries
        data = pd.to_numeric(df[feat], errors='coerce').dropna()

        row[f"{feat}_mean"] = f"{data.mean():.1f}"
        row[f"{feat}_std"] = f"{data.std():.1f}"
        row[f"{feat}_N"] = len(data)
        row[f"{feat}_range"] = f"{data.min():.1f}-{data.max():.1f}"

    summary.append(row)

summary_df = pd.DataFrame(summary)
print(summary_df.to_string(index=False))
summary_df.to_csv('spir_ppr_comparison.csv', index=False)


# Perform Kruskal–Wallis statistical test (non-parametric ANOVA)
print("\nStatistical Tests (Kruskal-Wallis)")
print("Feature    H-stat   p-value  Sig")

for feat in meaningful_features:
    groups_data = {}

    for name, df in datasets.items():
        data = pd.to_numeric(df[feat], errors='coerce').dropna()
        if data.std() > 0:
            groups_data[name] = data

    if len(groups_data) >= 2:
        h_stat, p_val = stats.kruskal(*groups_data.values())
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"{feat:<10} {h_stat:>6.1f}   {p_val:>7.4f}  {sig}")
    else:
        print(f"{feat:<10} {'-':>6}   {'-':>7}  [no variation]")


# Create boxplots comparing speech rate (SPIR) and pauses (PPR)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

spir_data = []
for name, df in datasets.items():
    data = pd.to_numeric(df['spir'], errors='coerce').dropna()
    spir_data.append(pd.DataFrame({'spir': data, 'group': name}))
spir_df = pd.concat(spir_data)
sns.boxplot(data=spir_df, x='group', y='spir', ax=ax1)
ax1.set_title('SPIR (Speech Rate)')
ax1.tick_params(axis='x', rotation=45)

ppr_data = []
for name, df in datasets.items():
    data = pd.to_numeric(df['ppr'], errors='coerce').dropna()
    ppr_data.append(pd.DataFrame({'ppr': data, 'group': name}))
ppr_df = pd.concat(ppr_data)
sns.boxplot(data=ppr_df, x='group', y='ppr', ax=ax2)
ax2.set_title('PPR (Pauses per Reading)')
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('spir_ppr_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nGenerated files:")
print("spir_ppr_comparison.csv")
print("spir_ppr_comparison.png")
