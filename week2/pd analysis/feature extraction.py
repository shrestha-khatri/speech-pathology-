
"""
Parkinson’s Disease Speech Feature Extraction Script

This script extracts acoustic speech features from WAV recordings of
participants diagnosed with Parkinson’s disease.

The pipeline performs:
• Automatic traversal of participant folders
• Prosodic feature extraction using Librosa
• Phonatory feature extraction using Praat (Parselmouth)
• Automatic task inference from filenames
• Feature aggregation into a CSV dataset
• Basic visualisation and descriptive statistics

Extracted feature groups:
Prosodic → duration, spir, ppr, n_pauses
Phonatory → hnr, jitter_local, jitter_rap, jitter_ppq5, shimmer_local, shimmer_apq3, shimmer_apq5

Outputs generated:
parkinsonPD_paper_features.csv
parkinsonPD_<feature>.png plots
"""

import pandas as pd
import numpy as np
import librosa
import parselmouth
from pathlib import Path
import os
import re
from tqdm import tqdm


print("Parkinson’s Disease Feature Extractor (28 subjects)")
print("=" * 70)

# Root directory containing Parkinson’s audio recordings
AUDIO_ROOT = Path(
    r"C:\Users\DELL\Codes\DH604\data\28 People with Parkinson's disease"
)


def infer_task(filename: str) -> str:
    """Infer speech task from filename using naming conventions."""
    fn = filename.lower()
    if re.search(r'b1|b2', fn):
        return 'simple'
    if re.search(r'pr', fn):
        return 'reading'
    if re.search(r'd1|d2', fn):
        return 'diadochokinetic'
    if re.search(r'v1|v2|vo|vu|va|ve', fn):
        return 'vowel'
    return 'other'


def extract_prosodic_librosa(file_path: Path):
    """Extract prosodic features using Librosa."""
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
        start = 0

        for i in range(len(silent)):
            if silent[i] and not in_pause:
                in_pause = True
                start = i
            elif not silent[i] and in_pause:
                dur = (i - start) * (hop_len / sr)
                if dur >= 0.25:
                    pauses.append(dur)
                in_pause = False

        n_pauses = len(pauses)
        total_pause = sum(pauses)

        spir = (n_pauses / duration) * 60 if duration > 0 else 0
        ppr = total_pause / duration if duration > 0 else 0

        return {'duration': duration, 'spir': spir, 'ppr': ppr, 'n_pauses': n_pauses}

    except:
        return {'duration': 0, 'spir': 0, 'ppr': 0, 'n_pauses': 0}


def extract_phonatory_praat(sound, pointProcess):
    """Extract phonatory voice quality features using Praat."""
    try:
        harmonicity = sound.to_harmonicity_cc(0.01, 75, 0.1, 1.0)
        hnr = harmonicity.get_mean(0, 0)
        harmonicity.remove()

        return {
            'hnr': float(hnr) if not np.isnan(hnr) else 0.0,
            'jitter_local': float(pointProcess.get_jitter_local(sound, 0, 0, 0.0001, 0.02, 1.3)),
            'jitter_rap': float(pointProcess.get_jitter_rap(sound, 0, 0, 0.0001, 0.02, 1.3)),
            'jitter_ppq5': float(pointProcess.get_jitter_ppq5(sound, 0, 0, 0.0001, 0.02, 1.3)),
            'shimmer_local': float(pointProcess.get_shimmer_local(sound, 0, 0, 0.0001, 0.02, 1.3)),
            'shimmer_apq3': float(pointProcess.get_shimmer_apq3(sound, 0, 0, 0.0001, 0.02, 1.3)),
            'shimmer_apq5': float(pointProcess.get_shimmer_apq5(sound, 0, 0, 0.0001, 0.02, 1.3))
        }

    except:
        return {
            'hnr': 0.0, 'jitter_local': 0.0, 'jitter_rap': 0.0,
            'jitter_ppq5': 0.0, 'shimmer_local': 0.0,
            'shimmer_apq3': 0.0, 'shimmer_apq5': 0.0
        }


# Main extraction loop
rows = []

for root, dirs, files in tqdm(os.walk(AUDIO_ROOT)):
    for f in files:
        if f.lower().endswith('.wav'):
            file_path = Path(root) / f

            prosodic = extract_prosodic_librosa(file_path)

            try:
                sound = parselmouth.Sound(str(file_path))
                pointProcess = sound.to_pointprocess(75, 500)
                phonatory = extract_phonatory_praat(sound, pointProcess)
                pointProcess.remove()
                sound.remove()
            except:
                phonatory = {
                    'hnr': 0.0, 'jitter_local': 0.0, 'jitter_rap': 0.0,
                    'jitter_ppq5': 0.0, 'shimmer_local': 0.0,
                    'shimmer_apq3': 0.0, 'shimmer_apq5': 0.0
                }

            row = {
                'filename': f,
                'subject_id': file_path.parent.name,
                'task': infer_task(f),
                'group': 'parkinsonPD'
            }

            row.update(prosodic)
            row.update(phonatory)
            rows.append(row)


# Save dataset
df_pd = pd.DataFrame(rows)
df_pd.to_csv('parkinsonPD_paper_features.csv', index=False)

print("Feature extraction completed")
print(f"Total files processed: {len(df_pd)}")
print(f"Total subjects found: {df_pd['subject_id'].nunique()}")


# Visualisation
import seaborn as sns
import matplotlib.pyplot as plt

feat_cols = ['spir', 'ppr', 'hnr', 'jitter_local', 'shimmer_local']
df_clean = df_pd.dropna(subset=feat_cols, how='all')

for feat in feat_cols:
    feat_df = df_clean.dropna(subset=[feat])
    if len(feat_df) >= 4:
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=feat_df, x='task', y=feat)
        plt.title(f'{feat.upper()} by Task (Parkinson’s, n={len(feat_df)})')
        plt.tight_layout()
        plt.savefig(f'parkinsonPD_{feat}.png', dpi=300, bbox_inches='tight')
        plt.show()

print("\nMean feature values by task (Parkinson’s):")
print(df_pd.groupby('task')[feat_cols].mean().round(3))

