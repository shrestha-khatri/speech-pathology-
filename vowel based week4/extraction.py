
"""
Task-Level Phonatory Feature Extraction Script

This script extracts phonatory voice features using Praat via Parselmouth
for three groups: youngHC, elderlyHC, and parkinsonPD.

Unlike earlier scripts, this version preserves the individual task labels
(VA1, VA2, B1, B2, PR1, D1, D2). These task labels are later used in the
task-based statistical analysis pipeline.

For every WAV file the script extracts
HNR, jitter (local, RAP, PPQ5), shimmer (local, APQ3, APQ5),
mean F0 and F0 standard deviation.

Output file
phonatory_task_grouped.csv
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call
import re


# Root folders for each participant group
YOUNG_ROOT     = Path(r"D:\Download\Codes\DH604\data\15 Young Healthy Control")
ELDERLY_ROOT   = Path(r"D:\Download\Codes\DH604\data\22 Elderly Healthy Control")
PARKINSON_ROOT = Path(r"D:\Download\Codes\DH604\data\28 People with Parkinson's disease")

# Output CSV where all extracted features will be stored
OUTPUT_CSV     = "phonatory_task_grouped.csv"


def safe_float(x):
    """
    Convert values safely to float.
    Any NaN or invalid value becomes 0.0.
    This prevents Praat failures from breaking the pipeline.
    """
    try:
        return float(x) if not np.isnan(float(x)) else 0.0
    except:
        return 0.0


def infer_task_type(filename: str) -> str:
    """
    Identify speech task using filename patterns.
    Each dataset follows naming conventions such as:
    VA1 VA2  vowel phonation
    B1  B2   monologue or breathing tasks
    PR1      reading task
    D1  D2   diadochokinetic syllable repetition
    """
    fn = filename.lower()

    if re.search(r"va1", fn): return "VA1"
    if re.search(r"va2", fn): return "VA2"
    if re.search(r"b1", fn):  return "B1"
    if re.search(r"b2", fn):  return "B2"
    if re.search(r"pr1|pr", fn): return "PR1"
    if re.search(r"d1", fn):  return "D1"
    if re.search(r"d2", fn):  return "D2"

    return "OTHER"


def extract_phonatory_parselmouth(file_path: Path) -> dict:
    """
    Extract phonatory features using Praat via Parselmouth.

    Extracted biomarkers
    HNR  Harmonics to Noise Ratio
    Jitter cycle to cycle frequency instability
    Shimmer cycle to cycle amplitude instability
    F0 mean and variability
    """
    try:
        # Load audio file
        sound = parselmouth.Sound(str(file_path))

        # Pitch estimation limits for human voice
        f0_min, f0_max = 75, 500

        # Extract pitch contour
        pitch = call(sound, "To Pitch", 0.0, f0_min, f0_max)

        # Keep only voiced frames
        f0_values = pitch.selected_array["frequency"]
        f0_values = f0_values[f0_values > 0]

        # Compute F0 statistics
        f0_mean = float(np.mean(f0_values)) if len(f0_values) > 0 else 0.0
        f0_std  = float(np.std(f0_values)) if len(f0_values) > 0 else 0.0

        # Convert pitch to glottal pulse representation
        pulses = call([sound, pitch], "To PointProcess (cc)")

        # Harmonicity object used for HNR computation
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, f0_min, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)

        # Jitter metrics quantify frequency instability
        jitter_local = call(pulses, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        jitter_rap   = call(pulses, "Get jitter (rap)",   0, 0, 0.0001, 0.02, 1.3)
        jitter_ppq5  = call(pulses, "Get jitter (ppq5)",  0, 0, 0.0001, 0.02, 1.3)

        # Shimmer metrics quantify amplitude instability
        shimmer_local = call([sound, pulses], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_apq3  = call([sound, pulses], "Get shimmer (apq3)",  0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_apq5  = call([sound, pulses], "Get shimmer (apq5)",  0, 0, 0.0001, 0.02, 1.3, 1.6)

        # Return all features in dictionary form
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

    # If Praat fails, return zeros so the pipeline never crashes
    except Exception as e:
        print(f"[ERROR] {file_path}: {e}")
        return {
            k: 0.0 for k in [
                "hnr","jitter_local","jitter_rap","jitter_ppq5",
                "shimmer_local","shimmer_apq3","shimmer_apq5",
                "f0_mean","f0_std"
            ]
        }


def process_dataset(root: Path, group_name: str) -> pd.DataFrame:
    """
    Walk through dataset directory and process all WAV files.
    Each speaker folder is scanned recursively.
    """
    rows = []

    for dirpath, dirnames, filenames in os.walk(root):
        speaker = Path(dirpath).name

        for fname in filenames:
            if not fname.lower().endswith(".wav"):
                continue

            # Identify task label from filename
            task = infer_task_type(fname)

            # Skip files that do not belong to known tasks
            if task == "OTHER":
                continue

            file_path = Path(dirpath) / fname
            print(f"[{group_name}/{speaker}] {fname} -> {task}")

            # Extract phonatory features
            feats = extract_phonatory_parselmouth(file_path)

            # Build row metadata
            row = {
                "speaker": speaker,
                "filename": fname,
                "group": group_name,
                "task": task
            }

            # Append extracted features
            row.update(feats)
            rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Starting task based phonatory feature extraction")

    # Process each group independently
    df_young   = process_dataset(YOUNG_ROOT, "youngHC")
    df_elderly = process_dataset(ELDERLY_ROOT, "elderlyHC")
    df_pd      = process_dataset(PARKINSON_ROOT, "parkinsonPD")

    # Combine all groups into one dataset
    df_all = pd.concat([df_young, df_elderly, df_pd], ignore_index=True)

    # Save final dataset used by statistical pipeline
    df_all.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved {len(df_all)} rows to {OUTPUT_CSV}")
    print("Task counts:")
    print(df_all.groupby(['task','group']).size().unstack(fill_value=0))

