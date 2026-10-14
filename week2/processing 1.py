"""
Cross-Group CPS Analysis and Simple Classification Script

This script combines reading-speed (Characters Per Second, CPS) data from:
• Young Healthy Controls
• Elderly Healthy Controls
• Parkinson’s Disease participants

The pipeline performs:
• Data loading from Excel sheets
• Cleaning and harmonising column formats
• Combining datasets into one master dataframe
• Visualisation of CPS and demographic distributions
• Mann–Whitney statistical testing (PD vs controls)
• Simple logistic-regression classification using CPS
• Export of combined dataset

Outputs:
cps_by_group.png
PRESENTATION_ALL_GRAPHS.png
SLIDE1_speech_rate.png
SLIDE2_age.png
PD_speech_features.csv
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# Load Excel files
yhc = pd.read_excel(r"C:\Users\DELL\Codes\DH604\15 Young Healthy Control\15 YHC.xlsx", sheet_name=0)
pd_df = pd.read_excel(r"C:\Users\DELL\Codes\DH604\22 Elderly Healthy Control\Tab 3.xlsx", sheet_name=0)
ehc = pd.read_excel(r"C:\Users\DELL\Codes\DH604\28 People with Parkinson's disease\TAB 5.xlsx", sheet_name=0)

print("Files loaded successfully")
print("YHC shape:", yhc.shape)
print("PD shape:", pd_df.shape)
print("EHC shape:", ehc.shape)


# Clean and standardise columns
yhc.columns = ["name", "surname", "sex", "age", "from", "time1", "CPS1", "time2", "CPS2", "time3", "CPS3"]
yhc = yhc[~yhc["name"].isna()]
yhc["group"] = "young_HC"
yhc["subject_id"] = ["YHC{:02d}".format(i+1) for i in range(len(yhc))]

pd_df.columns = ["name", "surname", "sex", "age", "from", "time1", "CPS1", "time2", "CPS2", "time3", "CPS3"]
pd_df = pd_df[~pd_df["name"].isna()]
pd_df["group"] = "PD"
pd_df["subject_id"] = ["PD{:02d}".format(i+1) for i in range(len(pd_df))]

ehc.columns = ["dummy", "name", "surname", "sex", "age", "time1", "CPS1", "time2", "CPS2", "time3", "CPS3"]
ehc = ehc.drop(columns=["dummy"])
ehc = ehc[~ehc["name"].isna()]
ehc["group"] = "elderly_HC"
ehc["subject_id"] = ["EHC{:02d}".format(i+1) for i in range(len(ehc))]


# Combine datasets
all_df = pd.concat([yhc, pd_df, ehc], ignore_index=True)
all_df = all_df[all_df["name"] != "name"]

for col in ["CPS1", "CPS2", "CPS3"]:
    all_df[col] = pd.to_numeric(all_df[col], errors='coerce')

all_df = all_df.dropna(subset=["CPS1", "CPS2", "CPS3"])

print("\nFinal subjects by group:")
print(all_df["group"].value_counts())


# Boxplot of CPS across groups and texts
cps_long = all_df.melt(
    id_vars=["subject_id", "group"],
    value_vars=["CPS1", "CPS2", "CPS3"],
    var_name="text",
    value_name="CPS"
)

plt.figure(figsize=(10, 6))
sns.boxplot(data=cps_long, x="group", y="CPS", hue="text")
plt.title("Characters per second (CPS) across groups")
plt.savefig("cps_by_group.png", dpi=300, bbox_inches='tight')
plt.show()


print("\nSUMMARY STATISTICS")
print(all_df["group"].value_counts())

cps_summary = all_df.groupby('group')[['CPS1','CPS2','CPS3']].agg(['mean','std']).round(2)
print(cps_summary)


# Compute mean CPS per subject
all_df["CPS_mean"] = all_df[["CPS1","CPS2","CPS3"]].mean(axis=1)


# Mann–Whitney tests (PD vs Controls)
from scipy.stats import mannwhitneyu

print("\nSTATISTICAL TESTS (PD vs Controls)")
for feature in ["CPS_mean","CPS1","CPS2","CPS3"]:
    pd_vals = all_df[all_df["group"]=="PD"][feature]
    hc_vals = all_df[all_df["group"].isin(["young_HC","elderly_HC"])][feature]
    stat,p = mannwhitneyu(pd_vals,hc_vals)
    print(f"{feature}: p={p:.3f}")


# Presentation plots
plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
sns.boxplot(data=cps_long,x="group",y="CPS",hue="text")
plt.title("CPS by Reading Text")

plt.subplot(2,2,2)
sns.boxplot(data=all_df,x="group",y="CPS_mean")
plt.title("Mean Speech Rate (CPS)")

plt.subplot(2,2,3)
sns.boxplot(data=all_df,x="group",y="age")
plt.title("Age by Group")

plt.subplot(2,2,4)
sns.violinplot(data=all_df,x="group",y="CPS_mean")
plt.title("CPS Distribution")

plt.tight_layout()
plt.savefig("PRESENTATION_ALL_GRAPHS.png",dpi=300,bbox_inches='tight')
plt.show()


# Logistic regression classifier (PD vs HC)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

print("\nSIMPLE CLASSIFIER (PD vs HC)")

df_clf = all_df.copy()
df_clf["label"] = (df_clf["group"]=="PD").astype(int)

X = df_clf[["CPS_mean"]].values
y = df_clf["label"].values

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression())
])

scores = cross_val_score(model,X,y,cv=5,scoring="accuracy")
print(f"CV Accuracy (CPS_mean only): {scores.mean():.1%} ± {scores.std():.1%}")


# Save combined dataset
all_df.to_csv("PD_speech_features.csv", index=False)
print("Files saved")
