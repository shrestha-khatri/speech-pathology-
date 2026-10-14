"""
Phonatory Feature Extraction for All Groups

This script extracts phonatory voice features from WAV recordings using
Praat via Parselmouth for three participant groups:
youngHC, elderlyHC, and parkinsonPD.

For every audio file the script extracts:
HNR, jitter (local, RAP, PPQ5), shimmer (local, APQ3, APQ5),
mean F0 and F0 standard deviation.

The script walks through three dataset folders, extracts features from
every WAV file, combines all rows into a single dataframe and exports it.

Output file:
phonatory_all_groups.csv
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call
import re


# Root folders for each group dataset
YOUNG_ROOT     = Path(r"D:\Download\Codes\DH604\data\01 Young HC")
ELDERLY_ROOT   = Path(r"D:\Download\Codes\DH604\data\22 Elderly Healthy Control")
PARKINSON_ROOT = Path(r"D:\Download\Codes\DH604\data\28 Parkinson")

# Output CSV file name
OUTPUT_CSV = "phonatory_all_groups.csv"


def infer_task(filename: str) -> str:
    """
    Infer speech task from filename using naming conventions.
    """
    fn = filename.lower()
    if re.search(r"b1|b2", fn): return "simple"
    if re.search(r"pr1|pr", fn): return "reading"
    if re.search(r"d1|d2", fn): return "diadochokinetic"
    if re.search(r"va1|va2", fn): return "vowel"
    return "other"


def safe_float(x) -> float:
    """
    Convert value to float safely and replace NaN or errors with 0.
    """
    try:
        x = float(x)
        if np.isnan(x):
            return 0.0
        return x
    except Exception:
        return 0.0


def extract_phonatory_parselmouth(file_path: Path) -> dict:
    """
    Extract phonatory features from a WAV file using Praat via Parselmouth.
    """
    try:
        sound = parselmouth.Sound(str(file_path))

        # Pitch extraction for F0 statistics
        f0_min, f0_max = 75, 500
        pitch = call(sound, "To Pitch", 0.0, f0_min, f0_max)
        f0_values = pitch.selected_array["frequency"]
        f0_values = f0_values[f0_values > 0]

        f0_mean = float(np.mean(f0_values)) if len(f0_values) > 0 else 0.0
        f0_std  = float(np.std(f0_values)) if len(f0_values) > 0 else 0.0

        # Extract glottal pulse locations
        pulses = call([sound, pitch], "To PointProcess (cc)")

        # Harmonic-to-noise ratio
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, f0_min, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)

        # Jitter measures
        jitter_local = call(pulses, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        jitter_rap   = call(pulses, "Get jitter (rap)",   0, 0, 0.0001, 0.02, 1.3)
        jitter_ppq5  = call(pulses, "Get jitter (ppq5)",  0, 0, 0.0001, 0.02, 1.3)

        # Shimmer measures
        shimmer_local = call([sound, pulses], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_apq3  = call([sound, pulses], "Get shimmer (apq3)",  0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_apq5  = call([sound, pulses], "Get shimmer (apq5)",  0, 0, 0.0001, 0.02, 1.3, 1.6)

        return {
            "hnr": safe_float(hnr),
            "jitter_local": safe_float(jitter_local),
            "jitter_rap": safe_float(jitter_rap),
            "jitter_ppq5": safe_float(jitter_ppq5),
            "shimmer_local": safe_float(shimmer_local),
            "shimmer_apq3": safe_float(shimmer_apq3),
            "shimmer_apq5": safe_float(shimmer_apq5),
            "f0_mean": safe_float(f0_mean),
            "f0_std": safe_float(f0_std),
        }

    except Exception as e:
        print(f"[ERROR] {file_path}: {e}")
        return {
            "hnr": 0.0,
            "jitter_local": 0.0,
            "jitter_rap": 0.0,
            "jitter_ppq5": 0.0,
            "shimmer_local": 0.0,
            "shimmer_apq3": 0.0,
            "shimmer_apq5": 0.0,
            "f0_mean": 0.0,
            "f0_std": 0.0,
        }


def process_dataset(root: Path, group_name: str) -> pd.DataFrame:
    """
    Walk through a dataset folder and extract features from all WAV files.
    """
    rows = []

    for dirpath, dirnames, filenames in os.walk(root):
        for fname in filenames:
            if not fname.lower().endswith(".wav"):
                continue

            file_path = Path(dirpath) / fname
            print(f"[{group_name}] Processing {fname}")

            feats = extract_phonatory_parselmouth(file_path)

            row = {
                "filename": fname,
                "folder": Path(dirpath).name,
                "group": group_name,
                "task": infer_task(fname),
            }

            row.update(feats)
            rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Starting phonatory feature extraction")

    df_young   = process_dataset(YOUNG_ROOT, "youngHC")
    df_elderly = process_dataset(ELDERLY_ROOT, "elderlyHC")
    df_pd      = process_dataset(PARKINSON_ROOT, "parkinsonPD")

    df_all = pd.concat([df_young, df_elderly, df_pd], ignore_index=True)
    df_all.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved {len(df_all)} rows to {OUTPUT_CSV}")
    print("Columns:", list(df_all.columns))
