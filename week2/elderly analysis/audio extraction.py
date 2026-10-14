"""
Elderly Healthy Control Speech Feature Extraction Script

This script extracts acoustic speech features from WAV recordings of
elderly healthy control participants.

The pipeline performs:
• Automatic traversal of participant folders
• Prosodic feature extraction using Librosa
• Phonatory feature extraction using Praat (via Parselmouth)
• Task inference from filenames
• Feature aggregation into a CSV dataset
• Basic visualisation and descriptive statistics

Extracted feature groups:

Prosodic (pause & speech rate)
    duration, spir, ppr, n_pauses

Phonatory (voice quality)
    hnr, jitter_local, jitter_rap, jitter_ppq5,
    shimmer_local, shimmer_apq3, shimmer_apq5

Outputs generated:
elderlyHC_paper_features.csv
elderlyHC_<feature>.png plots
"""

import pandas as pd
import numpy as np
import librosa
import parselmouth
from pathlib import Path
import os
import re
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from tqdm import tqdm


print("Elderly HC Features Extractor")


# Root folder containing elderly audio recordings
AUDIO_ROOT = Path(r"C:\Users\DELL\Codes\DH604\data\22 Elderly Healthy Control")


def infer_task(filename: str) -> str:
    """
    Infer speech task type from filename.
    Naming conventions determine task category.
    """
    fn_lower = filename.lower()
    if re.search(r'b1|b2', fn_lower):
        return 'simple'
    if re.search(r'pr1|pr', fn_lower):
        return 'reading'
    if re.search(r'd1|d2', fn_lower):
        return 'diadochokinetic'
    return 'other'


def extract_prosodic_librosa(file_path: Path):
    """
    Extract prosodic features using Librosa.

    SPIR: speech pauses per minute
    PPR : pause proportion ratio
    """
    try:
        y, sr = librosa.load(str(file_path), sr=22050)
        duration = len(y) / sr

        if duration < 0.5:
            return {'duration': duration, 'spir': 0.0, 'ppr': 0.0, 'n_pauses': 0}

        frame_len = int(0.025 * sr)
        hop_len = int(0.010 * sr)
        rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]

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

        return {'duration': duration, 'spir': spir, 'ppr': ppr, 'n_pauses': n_pauses}

    except:
        return {'duration': 0, 'spir': 0, 'ppr': 0, 'n_pauses': 0}


def extract_phonatory_praat(sound, pointProcess):
    """
    Extract phonatory voice quality features using Praat.
    Includes jitter, shimmer, and Harmonic-to-Noise Ratio.
    """
    try:
        harmonicity = sound.to_harmonicity_cc(0.01, 75, 0.1, 1.0)
        hnr = harmonicity.get_mean(0, 0)
        harmonicity.remove()

        localJitter = pointProcess.get_jitter_local(sound, 0, 0, 0.0001, 0.02, 1.3)
        rapJitter = pointProcess.get_jitter_rap(sound, 0, 0, 0.0001, 0.02, 1.3)
        ppq5Jitter = pointProcess.get_jitter_ppq5(sound, 0, 0, 0.0001, 0.02, 1.3)

        localShimmer = pointProcess.get_shimmer_local(sound, 0, 0, 0.0001, 0.02, 1.3)
        apq3Shimmer = pointProcess.get_shimmer_apq3(sound, 0, 0, 0.0001, 0.02, 1.3)
        apq5Shimmer = pointProcess.get_shimmer_apq5(sound, 0, 0, 0.0001, 0.02, 1.3)

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
        return {'hnr': 0.0, 'jitter_local': 0.0, 'jitter_rap': 0.0,
                'jitter_ppq5': 0.0, 'shimmer_local': 0.0,
                'shimmer_apq3': 0.0, 'shimmer_apq5': 0.0}


rows = []

for root, dirs, files in os.walk(AUDIO_ROOT):
    for f in files:
        if f.lower().endswith('.wav'):
            file_path = Path(root) / f
            print(f"Processing {f}")

            prosodic = extract_prosodic_librosa(file_path)

            try:
                sound = parselmouth.Sound(str(file_path))
                pointProcess = sound.to_pointprocess(75, 500)
                phonatory = extract_phonatory_praat(sound, pointProcess)
                pointProcess.remove()
                sound.remove()
            except:
                phonatory = {'hnr': 0.0, 'jitter_local': 0.0, 'jitter_rap': 0.0,
                             'jitter_ppq5': 0.0, 'shimmer_local': 0.0,
                             'shimmer_apq3': 0.0, 'shimmer_apq5': 0.0}

            row = {
                'filename': f,
                'folder': os.path.basename(root),
                'task': infer_task(f),
                'group': 'elderlyHC',
            }

            row.update(prosodic)
            row.update(phonatory)
            rows.append(row)


df = pd.DataFrame(rows)
df.to_csv('elderlyHC_paper_features.csv', index=False)
print(f"Saved {len(df)} rows")


feat_cols = ['spir', 'ppr', 'hnr', 'jitter_local', 'shimmer_local']
df_clean = df.dropna(subset=feat_cols, how='all')

for feat in feat_cols:
    feat_df = df_clean.dropna(subset=[feat])
    if len(feat_df) >= 4:
        plt.figure(figsize=(8,5))
        sns.boxplot(data=feat_df, x='task', y=feat)
        plt.title(f'{feat.upper()} by Task')
        plt.savefig(f'elderlyHC_{feat}.png', dpi=300, bbox_inches='tight')
        plt.show()

print(df.groupby('task')[feat_cols].mean().round(3))
