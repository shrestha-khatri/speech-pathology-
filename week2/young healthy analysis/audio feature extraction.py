"""
Young Healthy Control Feature Extraction Script

This script extracts speech features from WAV files for the Young Healthy
Control (youngHC) group.

Two categories of features are computed:

1. Prosodic features (Kovac-style):
   - duration
   - SPIR (speech pauses per minute)
   - PPR (pause proportion ratio)
   - n_pauses

2. Phonatory features (Favaro-style using Praat via Parselmouth):
   - HNR
   - jitter (local, RAP, PPQ5)
   - shimmer (local, APQ3, APQ5)

The script:
- Walks through all WAV files in the specified folder
- Extracts features per file
- Saves results to youngHC_paper_features.csv
- Generates boxplots for key features
- Prints mean feature values by task
"""

import pandas as pd
import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call
from pathlib import Path
import os
import re
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from tqdm import tqdm


print("PD Paper Features Extractor (Kovac + Favaro)")


# Path to Young Healthy Control audio files
AUDIO_ROOT = Path(r"C:\Users\DELL\Codes\DH604\data\15 Young Healthy Control")


def infer_task(filename: str) -> str:
    """
    Infer task type from filename using naming conventions.
    """
    fn_lower = filename.lower()
    if re.search(r'b1|b2', fn_lower): return 'simple'
    if re.search(r'pr1|pr', fn_lower): return 'reading'
    return 'other'


def extract_prosodic_librosa(file_path: Path):
    """
    Extract prosodic features using librosa.

    Computes:
    - duration (seconds)
    - SPIR (speech pauses per minute)
    - PPR (pause proportion ratio)
    - number of pauses
    """
    try:
        y, sr = librosa.load(str(file_path), sr=22050)
        duration = len(y) / sr

        if duration < 0.5:
            return {'duration': duration, 'spir': 0.0, 'ppr': 0.0, 'n_pauses': 0}

        frame_len = int(0.025 * sr)
        hop_len = int(0.010 * sr)

        rms = librosa.feature.rms(
            y=y,
            frame_length=frame_len,
            hop_length=hop_len
        )[0]

        thresh = rms.max() * 0.1
        silent = rms < thresh

        pauses = []
        in_pause = False

        for i in range(1, len(silent)):
            if silent[i] and not silent[i-1]:
                in_pause = True
            elif not silent[i] and in_pause:
                pause_dur = i * (hop_len / sr)
                if pause_dur >= 0.25:
                    pauses.append(pause_dur)
                in_pause = False

        n_pauses = len(pauses)
        total_pause = sum(pauses)

        spir = (n_pauses / duration) * 60 if duration > 0 else 0
        ppr = total_pause / duration if duration > 0 else 0

        return {
            'duration': duration,
            'spir': spir,
            'ppr': ppr,
            'n_pauses': n_pauses
        }

    except:
        return {'duration': 0, 'spir': 0, 'ppr': 0, 'n_pauses': 0}


def extract_phonatory_praat(sound, pointProcess):
    """
    Extract phonatory features using Praat via Parselmouth.

    Computes:
    - HNR
    - jitter (local, RAP, PPQ5)
    - shimmer (local, APQ3, APQ5)
    """
    try:
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)
        harmonicity.remove()

        localJitter = call([sound, pointProcess], "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        rapJitter = call([sound, pointProcess], "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
        ppq5Jitter = call([sound, pointProcess], "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)

        localShimmer = call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3)
        apq3Shimmer = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3)
        apq5Shimmer = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3)

        return {
            'hnr': float(hnr) if not np.isnan(hnr) else 0.0,
            'jitter_local': float(localJitter) if not np.isnan(localJitter) else 0.0,
            'jitter_rap': float(rapJitter) if not np.isnan(rapJitter) else 0.0,
            'jitter_ppq5': float(ppq5Jitter) if not np.isnan(ppq5Jitter) else 0.0,
            'shimmer_local': float(localShimmer) if not np.isnan(localShimmer) else 0.0,
            'shimmer_apq3': float(apq3Shimmer) if not np.isnan(apq3Shimmer) else 0.0,
            'shimmer_apq5': float(apq5Shimmer) if not np.isnan(apq5Shimmer) else 0.0,
        }

    except Exception as e:
        print(f"Praat error: {e}")
        return {
            'hnr': 0.0,
            'jitter_local': 0.0,
            'jitter_rap': 0.0,
            'jitter_ppq5': 0.0,
            'shimmer_local': 0.0,
            'shimmer_apq3': 0.0,
            'shimmer_apq5': 0.0
        }


rows = []

for root, dirs, files in os.walk(AUDIO_ROOT):
    for f in files:
        if f.lower().endswith('.wav'):
            file_path = Path(root) / f
            print(f"Processing {f}")

            prosodic = extract_prosodic_librosa(file_path)

            try:
                sound = parselmouth.Sound(str(file_path))
                pointProcess = call(sound, "To PointProcess (periodic, cc)", 75, 500)
                phonatory = extract_phonatory_praat(sound, pointProcess)
                pointProcess.remove()
                sound.remove()
            except:
                phonatory = {
                    'hnr': 0.0,
                    'jitter_local': 0.0,
                    'jitter_rap': 0.0,
                    'jitter_ppq5': 0.0,
                    'shimmer_local': 0.0,
                    'shimmer_apq3': 0.0,
                    'shimmer_apq5': 0.0
                }

            row = {
                'filename': f,
                'folder': os.path.basename(root),
                'task': infer_task(f),
                'group': 'youngHC'
            }

            row.update(prosodic)
            row.update(phonatory)
            rows.append(row)


df = pd.DataFrame(rows)
df.to_csv('youngHC_paper_features.csv', index=False)

print(f"Saved {len(df)} rows to youngHC_paper_features.csv")


feat_cols = ['spir', 'ppr', 'hnr', 'jitter_local', 'shimmer_local']
df_clean = df.dropna(subset=feat_cols, how='all')

for feat in feat_cols:
    feat_df = df_clean.dropna(subset=[feat])
    if len(feat_df) >= 4:
        plt.figure(figsize=(8,5))
        sns.boxplot(data=feat_df, x='task', y=feat)
        plt.title(f'{feat.upper()} by Task (Young HC)')
        plt.savefig(f'{feat}.png', dpi=300, bbox_inches='tight')
        plt.show()


print(df.groupby('task')[feat_cols].mean().round(3))
