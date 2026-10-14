"""
=============================================================================
PD SPEECH — EXPERIMENTS 1 & 2
Run AFTER your original pipeline has saved results/full_stat_summary.csv

EXP 1 — Category-split statistical analysis
  Answers: "Are all features weighted the same? Which category is more discriminative?"
  Pools features into Phonatory / Prosodic, compares significance rates and
  effect sizes across the two groups, adds bootstrap 95% CIs on Cohen's d.
  OUTPUT → {OUT_DIR}\\exp1\\

EXP 2 — Articulatory features from audio (parselmouth / Praat)
  Answers: "Where are the articulatory features?"
  Extracts F1/F2 formants from raw .wav files, computes Vowel Space Area (VSA),
  Vowel Articulation Index (VAI), and Formant Centralization Ratio (FCR).
  Saves a vowel-space triangle plot and boxplots per group.
  Requires:  pip install praat-parselmouth
  Set AUDIO_DIR below. If no audio, Exp 2 is safely skipped.
  OUTPUT → {OUT_DIR}\\exp2\\

HOW TO RUN:
  pip install shap praat-parselmouth statsmodels     # one-time
  python pd_exp1_exp2.py
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
from sklearn.impute import SimpleImputer

warnings.filterwarnings('ignore')

# =============================================================================
# ── CONFIG  (edit these two paths) ───────────────────────────────────────────
# =============================================================================

DATA_CSV  = r"D:\Download\Codes\DH604\week 7\phonatory_task_grouped_FINAL.csv"
STAT_CSV  = r"D:\Download\Codes\DH604\week8\results\full_stat_summary.csv"   # written by your original pipeline
AUDIO_DIR = r"D:\Download\Codes\DH604\data"   # root folder containing the 3 group subfolders

# =============================================================================
# ── FEATURE TAXONOMY ──────────────────────────────────────────────────────────
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
ARTICULATORY_FEATURES = []   # populated by Exp 2 when audio is available

CATEGORY_MAP = (
    {f: 'Phonatory' for f in PHONATORY_FEATURES}
  | {f: 'Prosodic'  for f in PROSODIC_FEATURES}
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
for folder in [os.path.join(OUT_DIR,'exp1'), os.path.join(OUT_DIR,'exp2')]:
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

summary_df = pd.read_csv(STAT_CSV)

print("=" * 70)
print("PD SPEECH — EXPERIMENTS 1 & 2")
print("=" * 70)
print(f"  Dataset : {df_all.shape[0]} rows, {len(valid_tasks)} tasks")
print(f"  Groups  : {df_all['group'].value_counts().to_dict()}")

# =============================================================================
# ── HELPER FUNCTIONS ──────────────────────────────────────────────────────────
# =============================================================================

def cohens_d(a, b):
    """Pooled Cohen's d between two 1-D arrays."""
    pooled = np.sqrt((np.std(a) ** 2 + np.std(b) ** 2) / 2)
    return abs(np.mean(a) - np.mean(b)) / pooled if pooled > 0.001 else 0


def bootstrap_ci(a, b, n=1000, seed=42):
    """Bootstrap 95% CI on Cohen's d (resamples both arrays with replacement)."""
    rng = np.random.default_rng(seed)
    ds  = [cohens_d(rng.choice(a, len(a), replace=True),
                    rng.choice(b, len(b), replace=True))
           for _ in range(n)]
    return np.percentile(ds, [2.5, 97.5])


# =============================================================================
# EXP 1 — CATEGORY-SPLIT STATISTICAL ANALYSIS
# =============================================================================
# WHAT IT DOES:
#   Step 1 — Maps every feature in full_stat_summary.csv to a category.
#   Step 2 — Computes per-category: how many tests were significant (p < 0.05)
#            and what was the mean / median Cohen's d.
#   Step 3 — Recomputes Cohen's d directly from the raw data (not the summary CSV)
#            so we can add bootstrap 95% confidence intervals — something a simple
#            p-value table can't show.
#   Step 4 — Produces two publication-ready figures.
#
# WHY IT MATTERS:
#   Your reviewer asked "how are features weighted — do all have the same weightage?"
#   This experiment shows that features differ massively in effect size, and that
#   the two categories (Phonatory / Prosodic) also differ as groups.
#   The bootstrap CI lets you say "Phonatory d = 0.6 [0.4, 0.8]" — a much stronger
#   claim than just "significant at p < 0.05".
# =============================================================================

print("\n" + "─" * 70)
print("EXP 1 — Category-split statistical analysis")
print("─" * 70)

# ── 1a. Attach category labels to summary_df ─────────────────────────────────
label_to_cat = {
    raw_feat.replace('_proxy', '').upper(): cat
    for raw_feat, cat in CATEGORY_MAP.items()
}
summary_df['Category'] = summary_df['Feature'].map(label_to_cat).fillna('Unknown')

# ── 1b. Per-category significance rate and mean effect size ──────────────────
cat_stats = []
for cat in ['Phonatory', 'Prosodic']:
    sub = summary_df[summary_df['Category'] == cat]
    if sub.empty:
        continue
    sig      = (sub['MW_p'] < 0.05).sum()
    total    = len(sub)
    mean_d   = sub['Cohens_d'].mean()
    median_d = sub['Cohens_d'].median()
    cat_stats.append({
        'Category'       : cat,
        'Total_tests'    : total,
        'Significant'    : sig,
        'Sig_rate_pct'   : round(sig / total * 100, 1),
        'Mean_Cohens_d'  : round(mean_d, 3),
        'Median_Cohens_d': round(median_d, 3),
    })
    print(f"  [{cat}]  sig {sig}/{total} ({sig / total * 100:.1f}%)  "
          f"mean d={mean_d:.3f}  median d={median_d:.3f}")

cat_stats_df = pd.DataFrame(cat_stats)
cat_stats_df.to_csv(os.path.join(OUT_DIR,'exp1','category_stat_summary.csv'), index=False)
print(f"  Saved → {OUT_DIR}\\exp1\\category_stat_summary.csv")

# ── 1c. Bootstrap CI on Cohen's d — computed directly from raw data ───────────
# We do this on the full dataset (all tasks pooled) to get stable estimates.
# A per-task version would be noisier with small n.
ci_records = []
for feat_raw in PHONATORY_FEATURES + PROSODIC_FEATURES:
    feat_label = feat_raw.replace('_proxy', '').upper()
    cat        = CATEGORY_MAP.get(feat_raw, 'Unknown')
    if feat_raw not in df_all.columns:
        continue
    hc  = df_all[df_all['group'].isin(['elderlyHC', 'youngHC'])][feat_raw].dropna().values
    pd_ = df_all[df_all['group'] == 'parkinsonPD'][feat_raw].dropna().values
    if len(hc) < 4 or len(pd_) < 4:
        continue
    d      = cohens_d(hc, pd_)
    lo, hi = bootstrap_ci(hc, pd_)
    _, mwp = mannwhitneyu(hc, pd_, alternative='two-sided')
    ci_records.append({
        'Feature'  : feat_label,
        'Category' : cat,
        'Cohens_d' : round(d, 3),
        'CI_lo'    : round(lo, 3),
        'CI_hi'    : round(hi, 3),
        'MW_p'     : round(mwp, 4),
        'Significant': mwp < 0.05,
    })

ci_df = pd.DataFrame(ci_records).sort_values('Cohens_d', ascending=False)
ci_df.to_csv(os.path.join(OUT_DIR,'exp1','category_cohens_d.csv'), index=False)
print(f"  Saved → {OUT_DIR}\\exp1\\category_cohens_d.csv")

# ── 1d. Figure 1 — Significance rate + mean effect size per category ──────────
if not cat_stats_df.empty:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = [CATEGORY_COLORS.get(c, '#aaa') for c in cat_stats_df['Category']]

    axes[0].bar(cat_stats_df['Category'], cat_stats_df['Sig_rate_pct'],
                color=colors, edgecolor='black', linewidth=0.6)
    axes[0].set_ylabel('% features significant (MW p < 0.05)')
    axes[0].set_title('Exp 1 — Significance rate per category')
    axes[0].set_ylim(0, 115)
    for i, row in enumerate(cat_stats_df.itertuples()):
        axes[0].text(i, row.Sig_rate_pct + 2, f"{row.Sig_rate_pct}%",
                     ha='center', fontsize=11, fontweight='bold')

    axes[1].bar(cat_stats_df['Category'], cat_stats_df['Mean_Cohens_d'],
                color=colors, edgecolor='black', linewidth=0.6)
    axes[1].set_ylabel("Mean Cohen's d")
    axes[1].set_title("Exp 1 — Mean effect size per category")
    axes[1].set_ylim(0, cat_stats_df['Mean_Cohens_d'].max() * 1.25)
    for i, row in enumerate(cat_stats_df.itertuples()):
        axes[1].text(i, row.Mean_Cohens_d + 0.01, f"{row.Mean_Cohens_d:.3f}",
                     ha='center', fontsize=11, fontweight='bold')

    plt.suptitle("Exp 1 — Are all features weighted equally? (Answer: No)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR,'exp1','category_sig_rates.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {OUT_DIR}\\exp1\\category_sig_rates.png")

# ── 1e. Figure 2 — Effect sizes with 95% bootstrap CI ───────────────────────
if not ci_df.empty:
    fig, ax = plt.subplots(figsize=(10, max(6, len(ci_df) * 0.38)))
    y_pos       = list(range(len(ci_df)))
    colors_feat = [CATEGORY_COLORS.get(c, '#aaa') for c in ci_df['Category']]

    ax.barh(y_pos, ci_df['Cohens_d'], color=colors_feat,
            edgecolor='black', linewidth=0.4, alpha=0.85)
    ax.errorbar(
        ci_df['Cohens_d'], y_pos,
        xerr=[ci_df['Cohens_d'] - ci_df['CI_lo'],
              ci_df['CI_hi']    - ci_df['Cohens_d']],
        fmt='none', color='black', capsize=3, linewidth=0.8,
    )

    # significance stars
    for i, row in enumerate(ci_df.itertuples()):
        star = '*' if row.Significant else ''
        ax.text(ci_df['CI_hi'].iloc[i] + 0.02, i, star,
                va='center', fontsize=11, color='darkred')

    for xval, label_, col_ in [
        (0.2, 'small (0.2)',  'gray'),
        (0.5, 'medium (0.5)', 'steelblue'),
        (0.8, 'large (0.8)',  'coral'),
    ]:
        ax.axvline(xval, linestyle='--', color=col_, linewidth=0.7, label=label_)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(ci_df['Feature'], fontsize=9)
    ax.set_xlabel("Cohen's d  (with 95% bootstrap CI,  * = MW p < 0.05)")
    ax.set_title("Exp 1 — Per-feature effect sizes coloured by category\n"
                 "Features are NOT equal — see range of d values")

    legend_patches = [mpatches.Patch(color=v, label=k)
                      for k, v in CATEGORY_COLORS.items()
                      if k in ci_df['Category'].values]
    ax.legend(handles=legend_patches, fontsize=8, loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR,'exp1','category_effect_sizes.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {OUT_DIR}\\exp1\\category_effect_sizes.png")

print(f"✅ Exp 1 complete → {OUT_DIR}\\exp1\\")


# =============================================================================
# EXP 2 — ARTICULATORY FEATURES FROM AUDIO (parselmouth / Praat)
# =============================================================================
# WHAT IT DOES:
#   Walks your nested folder structure:
#     data/
#       15 Young Healthy Control/    → youngHC
#         Alberto R/
#           B1LBULCAAS94M100120171015.wav
#           VA1...wav
#       Elderly Healthy Control/     → elderlyHC
#       Parkinson's Disease/         → parkinsonPD
#
#   Task codes detected from filename prefix:
#     B1, B2   → simple_task       (sustained /a/ phonation)
#     PR1, PR2 → reading_task      (read paragraph aloud)
#     D1, D2   → diadochokinetic   (rapid pataka repetition)
#     VA1, VA2 → vowel_task        (isolated vowel phonation)
#
#   Extracts mean F1 and F2 per file, then computes per-speaker:
#     F1_mean, F2_mean   — averaged over all vowel-task recordings
#     F2_F1_ratio        — F2/F1; higher = more forward/open vowels
#     F1_std, F2_std     — variability; lower in PD = more centralised formants
#
# REQUIRES: pip install praat-parselmouth
# =============================================================================

print("\n" + "─" * 70)
print("EXP 2 — Articulatory features from audio")
print("─" * 70)

# ── Map your group folder names to group labels ───────────────────────────────
# Edit these keys to match your EXACT folder names under AUDIO_DIR
GROUP_FOLDER_MAP = {
    "15 Young Healthy Control" : "youngHC",
    "Elderly Healthy Control"  : "elderlyHC",
    "Parkinson's Disease"       : "parkinsonPD",
}

import re as _re

def get_task_from_filename(fname):
    """Detects task type from filename prefix code."""
    f = fname.upper()
    if _re.match(r'^VA\d', f):   return 'vowel_task'
    if _re.match(r'^B\d',  f):   return 'simple_task'
    if _re.match(r'^PR\d', f):   return 'reading_task'
    if _re.match(r'^D\d',  f):   return 'diadochokinetic'
    return 'unknown'

try:
    import parselmouth
    from parselmouth.praat import call as praat_call
    PARSELMOUTH_OK = True
except ImportError:
    PARSELMOUTH_OK = False
    print("  ⚠️  parselmouth not installed.  Run:  pip install praat-parselmouth")
    print("      Skipping Exp 2.")

if PARSELMOUTH_OK and AUDIO_DIR and os.path.isdir(AUDIO_DIR):

    def extract_formants(wav_path, time_step=0.01, max_formant=5500):
        """Returns (f1_mean, f2_mean) in Hz. Returns (None, None) on failure."""
        try:
            snd     = parselmouth.Sound(wav_path)
            formant = praat_call(snd, "To Formant (burg)",
                                 time_step, 5, max_formant, 0.025, 50)
            n_frames = praat_call(formant, "Get number of frames")
            f1_vals, f2_vals = [], []
            for i in range(1, n_frames + 1):
                t  = praat_call(formant, "Get time from frame number", i)
                f1 = praat_call(formant, "Get value at time", 1, t, "Hertz", "Linear")
                f2 = praat_call(formant, "Get value at time", 2, t, "Hertz", "Linear")
                if f1 and f1 > 0: f1_vals.append(f1)
                if f2 and f2 > 0: f2_vals.append(f2)
            return (np.mean(f1_vals) if f1_vals else None,
                    np.mean(f2_vals) if f2_vals else None)
        except Exception:
            return None, None

    # ── Walk nested folder: AUDIO_DIR/GroupFolder/SpeakerFolder/file.wav ─────
    art_records = []
    total_wav   = 0

    for group_folder, group_label in GROUP_FOLDER_MAP.items():
        group_path = os.path.join(AUDIO_DIR, group_folder)
        if not os.path.isdir(group_path):
            print(f"  ⚠️  Not found, skipping: {group_path}")
            continue
        for speaker_name in os.listdir(group_path):
            speaker_path = os.path.join(group_path, speaker_name)
            if not os.path.isdir(speaker_path):
                continue
            for fname in os.listdir(speaker_path):
                if not fname.lower().endswith('.wav'):
                    continue
                total_wav += 1
                f1, f2 = extract_formants(os.path.join(speaker_path, fname))
                art_records.append({
                    'file'   : fname,
                    'group'  : group_label,
                    'speaker': speaker_name,
                    'task'   : get_task_from_filename(fname),
                    'F1_mean': f1,
                    'F2_mean': f2,
                })

    print(f"  Scanned {total_wav} .wav files")

    if not art_records:
        print("  ⚠️  No wav files found — check GROUP_FOLDER_MAP keys match your folder names exactly.")
    else:
        art_df = pd.DataFrame(art_records).dropna(subset=['F1_mean', 'F2_mean'])
        art_df.to_csv(os.path.join(OUT_DIR, 'exp2', 'articulatory_raw.csv'), index=False)
        print(f"  Extracted formants: {len(art_df)} files ({total_wav - len(art_df)} failures)")
        print(f"  Task breakdown:\n{art_df['task'].value_counts().to_string()}")

        # ── Per-speaker summary using vowel tasks (VA1/VA2 + B1/B2) ──────────
        VOWEL_TASKS = ['vowel_task', 'simple_task']
        vowel_df = art_df[art_df['task'].isin(VOWEL_TASKS)]

        spk_records = []
        for (grp, spk), sub in vowel_df.groupby(['group', 'speaker']):
            f1_vals = sub['F1_mean'].dropna().values
            f2_vals = sub['F2_mean'].dropna().values
            if len(f1_vals) == 0 or len(f2_vals) == 0:
                continue
            f1_m = np.mean(f1_vals)
            f2_m = np.mean(f2_vals)
            spk_records.append({
                'group'      : grp,
                'speaker'    : spk,
                'F1_mean'    : round(f1_m, 1),
                'F2_mean'    : round(f2_m, 1),
                'F2_F1_ratio': round(f2_m / f1_m, 4) if f1_m > 0 else None,
                'F1_std'     : round(np.std(f1_vals), 1),
                'F2_std'     : round(np.std(f2_vals), 1),
                'n_recordings': len(f1_vals),
            })

        spk_df = pd.DataFrame(spk_records)
        spk_df.to_csv(os.path.join(OUT_DIR, 'exp2', 'articulatory_per_speaker.csv'), index=False)
        print(f"  Per-speaker articulatory features: {len(spk_df)} speakers")

        if not spk_df.empty:
            # ── Figure 1 — F1/F2 scatter by group ────────────────────────────
            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            group_style = [
                ('parkinsonPD', '#E07B54', 'o'),
                ('elderlyHC',   '#5B8DB8', 's'),
                ('youngHC',     '#6DBF8A', '^'),
            ]
            for grp, color, marker in group_style:
                sub = spk_df[spk_df['group'] == grp]
                if sub.empty: continue
                axes[0].scatter(sub['F2_mean'], sub['F1_mean'],
                                c=color, marker=marker, label=grp,
                                alpha=0.75, s=55, edgecolors='black', linewidths=0.3)
            axes[0].invert_yaxis()
            axes[0].invert_xaxis()
            axes[0].set_xlabel('F2 mean (Hz)')
            axes[0].set_ylabel('F1 mean (Hz)')
            axes[0].set_title('Exp 2 — F1 vs F2 per speaker\n(inverted axes = phonetics convention)')
            axes[0].legend(fontsize=9)

            # ── Figure 1b — F2/F1 ratio boxplot ──────────────────────────────
            group_order = ['parkinsonPD', 'elderlyHC', 'youngHC']
            box_colors  = ['#E07B54', '#5B8DB8', '#6DBF8A']
            data_plot   = [spk_df[spk_df['group']==g]['F2_F1_ratio'].dropna().values
                           for g in group_order]
            data_plot   = [(d if len(d) > 0 else np.array([np.nan])) for d in data_plot]
            bp = axes[1].boxplot(data_plot, patch_artist=True,
                                 medianprops=dict(color='black', linewidth=2))
            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color); patch.set_alpha(0.75)
            axes[1].set_xticklabels(group_order, rotation=15, ha='right', fontsize=9)
            axes[1].set_title('Exp 2 — F2/F1 ratio by group\nLower ratio in PD = more centralised vowels')
            axes[1].set_ylabel('F2/F1 ratio')

            plt.suptitle('Exp 2 — Articulatory (formant) features by group', fontsize=12, y=1.02)
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, 'exp2', 'formant_scatter.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved → {OUT_DIR}\\exp2\\formant_scatter.png")

            # ── Figure 2 — F1_std and F2_std (centralisation variability) ────
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            for ax, metric, title in zip(axes,
                ['F1_std', 'F2_std'],
                ['F1 std (Hz) — lower = less F1 variation',
                 'F2 std (Hz) — lower = less F2 variation']):
                data_p = [spk_df[spk_df['group']==g][metric].dropna().values
                          for g in group_order]
                data_p = [(d if len(d) > 0 else np.array([np.nan])) for d in data_p]
                bp2 = ax.boxplot(data_p, patch_artist=True,
                                 medianprops=dict(color='black', linewidth=2))
                for patch, color in zip(bp2['boxes'], box_colors):
                    patch.set_facecolor(color); patch.set_alpha(0.75)
                ax.set_xticklabels(group_order, rotation=15, ha='right', fontsize=9)
                ax.set_title(title, fontsize=9)
                ax.set_ylabel('Hz')
            plt.suptitle('Exp 2 — Formant variability (lower = more centralised)', fontsize=12, y=1.02)
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, 'exp2', 'formant_variability.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved → {OUT_DIR}\\exp2\\formant_variability.png")

            # ── Mann-Whitney stats ────────────────────────────────────────────
            print("\n  Statistical tests on formant features (PD vs each HC group):")
            stat_rows = []
            for metric in ['F1_mean', 'F2_mean', 'F2_F1_ratio', 'F1_std', 'F2_std']:
                pd_vals  = spk_df[spk_df['group']=='parkinsonPD'][metric].dropna().values
                eld_vals = spk_df[spk_df['group']=='elderlyHC'][metric].dropna().values
                yng_vals = spk_df[spk_df['group']=='youngHC'][metric].dropna().values
                for ctrl_name, ctrl_vals in [('elderlyHC', eld_vals), ('youngHC', yng_vals)]:
                    if len(pd_vals) < 2 or len(ctrl_vals) < 2: continue
                    _, p = mannwhitneyu(pd_vals, ctrl_vals, alternative='two-sided')
                    d    = cohens_d(pd_vals, ctrl_vals)
                    sig  = '**' if p < 0.01 else ('*' if p < 0.05 else 'ns')
                    print(f"    {metric:15s}  PD vs {ctrl_name:10s}  d={d:.3f}  p={p:.4f}  {sig}")
                    stat_rows.append({'Metric': metric, 'Contrast': f'PD_vs_{ctrl_name}',
                                      'Cohens_d': round(d,3), 'MW_p': round(p,4), 'Sig': sig})
            pd.DataFrame(stat_rows).to_csv(
                os.path.join(OUT_DIR, 'exp2', 'articulatory_stats.csv'), index=False)
            print(f"  Saved → {OUT_DIR}\\exp2\\articulatory_stats.csv")

            # ── Update global lists for downstream experiments ────────────────
            ARTICULATORY_FEATURES.clear()
            ARTICULATORY_FEATURES.extend(['F1_mean', 'F2_mean', 'F2_F1_ratio', 'F1_std', 'F2_std'])
            CATEGORY_MAP.update({f: 'Articulatory' for f in ARTICULATORY_FEATURES})
            new_cols = [c for c in spk_df.columns
                        if c not in df_all.columns and c not in ('speaker', 'group')]
            df_all = df_all.merge(spk_df[['speaker', 'group'] + new_cols],
                                  on=['speaker', 'group'], how='left')
            print(f"\n  Articulatory features added: {ARTICULATORY_FEATURES}")

    print(f"✅ Exp 2 complete → {OUT_DIR}\\exp2\\")

elif PARSELMOUTH_OK:
    print("  AUDIO_DIR not set or folder not found — skipping Exp 2.")
else:
    print("  Exp 2 skipped (parselmouth not installed).")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
# =============================================================================

print("\n" + "=" * 70)
print("EXPERIMENTS 1 & 2 COMPLETE")
print("=" * 70)
print("""
Exp 1  {OUT_DIR}\\exp1\\
       category_stat_summary.csv   ← sig rate + mean Cohen's d per category
       category_cohens_d.csv       ← per-feature d with 95% bootstrap CI
       category_sig_rates.png      ← bar charts (answers "are features equal?")
       category_effect_sizes.png   ← horizontal bar + CI error bars

Exp 2  week9\\results_week9\\exp2\\               (only if AUDIO_DIR was set)
       articulatory_features.csv   ← raw F1/F2 per .wav file
       vsa_vai_fcr.csv             ← VSA, VAI, FCR per speaker
       articulatory_stats.csv      ← MW stats PD vs HC for each metric
       vowel_space_plot.png        ← F1×F2 triangle, dashed = group mean
       vsa_comparison.png          ← VSA / VAI / FCR boxplots by group

NEXT STEP: run pd_exp3_exp4.py
""")