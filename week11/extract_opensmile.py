# ================================
# extract_opensmile.py
# ================================

import os
import pandas as pd
import opensmile
from tqdm import tqdm


def extract_metadata(file_path):
    """
    Handles both structures:

    Young/Elderly:
    data/
        15 Young Healthy Control/
            Alberto R/
                B1.wav

    PD:
    data/
        28 People with Parkinson's disease/
            1-5/
                Domenico C/
                    B1.wav
    """
    parts = file_path.replace("\\", "/").split("/")

    filename   = parts[-1]
    subject_id = parts[-2]
    task       = filename.split(".")[0]

    # Search all parts for group keyword — works for any depth
    group = "Unknown"
    for part in parts:
        if "Young Healthy" in part:
            group = "Young"
            break
        elif "Elderly Healthy" in part:
            group = "Elderly"
            break
        elif "Parkinson" in part:
            group = "PD"
            break

    return subject_id, group, task, filename


def extract_egemaps(audio_root):

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    all_rows    = []
    audio_files = []

    for root, _, files in os.walk(audio_root):
        for f in files:
            if f.lower().endswith(".wav"):
                audio_files.append(os.path.join(root, f))

    print(f"Total .wav files found: {len(audio_files)}")

    for path in tqdm(audio_files):
        try:
            subject_id, group, task, file_id = extract_metadata(path)

            features_df   = smile.process_file(path)
            features_df   = features_df.reset_index(drop=True)
            features_dict = features_df.iloc[0].to_dict()

            row = {
                "subject_id": subject_id,
                "group":      group,
                "task":       task,
                "file_id":    file_id,
            }
            row.update(features_dict)
            all_rows.append(row)

        except Exception as e:
            print(f"ERROR: {path} -> {e}")

    df = pd.DataFrame(all_rows)

    print(f"\nExtraction complete. Shape: {df.shape}")
    print(f"Groups found:    {df['group'].value_counts().to_dict()}")
    print(f"Unique subjects: {df['subject_id'].nunique()}")

    # Warn if any unknown groups
    unknown = df[df["group"] == "Unknown"]
    if len(unknown) > 0:
        print(f"\nWARNING: {len(unknown)} files could not be assigned a group:")
        print(unknown[["subject_id", "file_id"]].head(10))

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-root", required=True,
                        help="Root data folder containing the 3 group subfolders")
    parser.add_argument("--out", default="egemaps.csv")
    args = parser.parse_args()

    # Delete old CSV if exists
    if os.path.exists(args.out):
        os.remove(args.out)
        print(f"Deleted old {args.out}")

    df = extract_egemaps(args.audio_root)
    df.to_csv(args.out, index=False)
    print(f"\nSaved → {args.out}")