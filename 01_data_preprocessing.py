"""
01_data_preprocessing.py
========================

Description:
    This script handles the ingestion and preprocessing of raw font pixel data.
    It is designed to filter specific font classes (Ranks 10-30 by frequency/size)
    to create a challenging "middle-rank" classification dataset.

Key Operations:
    1.  **Ingestion**: Reads raw CSV files from `data/inputs/`.
    2.  **Filtering**: Sorts fonts by file size and selects the slice [10:30].
        - Top 10 (Most frequent) -> Ignored (too easy/common).
        - Ranks 10-30 -> Selected as target classes.
        - Rest -> Lumped into a single "Other" class (optional usage).
    3.  **Reshaping**: Converts flattened 400-pixel vectors into (20x20) 2D images.
    4.  **Normalization**: Scales pixel values from [0, 255] to [0, 1].
    5.  **Serialization**: Saves tensors (X, y) and class names to `data/processed_fonts.pt`.

Usage:
    python 01_data_preprocessing.py
"""

import os

import glob
import pandas as pd
import numpy as np
import torch
from collections import Counter

# --- CONFIGURATION ---
RAW_DATA_DIR = "data/inputs"
OUTPUT_FILE = "data/processed_fonts.pt"

# TARGET SLICE: Fonts ranked 20th to 30th
# Python uses 0-based indexing, so:
# 0-20 would be the top 20.
# 20-30 are the next 10 (Rank 21 to Rank 30).
START_RANK = 10
END_RANK = 30


def preprocess_dataset():
    print(f"Starting pre-processing from {RAW_DATA_DIR}...")
    print(
        f"Target Slice: Ranks {START_RANK} to {END_RANK} (ignoring Top {START_RANK})."
    )

    # 1. Identify Classes by File Size
    csv_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {RAW_DATA_DIR}!")

    # Sort by size (Largest to Smallest)
    files_with_size = [(f, os.path.getsize(f)) for f in csv_files]
    sorted_files = sorted(files_with_size, key=lambda x: x[1], reverse=True)

    # --- NEW SLICING LOGIC ---
    # 1. The "Selected" Middle Slice (The classes we want to predict)
    selected_files = [f[0] for f in sorted_files[START_RANK:END_RANK]]

    # 2. The "Other" Pile (Everything else)
    # Includes the massive Top 20 (skipped) AND the tiny tail (after 30)
    skipped_top = [f[0] for f in sorted_files[:START_RANK]]
    skipped_bottom = [f[0] for f in sorted_files[END_RANK:]]
    other_files = skipped_top + skipped_bottom

    # Get Class Names
    class_names = [os.path.splitext(os.path.basename(f))[0] for f in selected_files]
    class_names.append("Other")  # The final class is 'Other'

    print(f"\n Selected Classes (Rank {START_RANK}-{END_RANK}):")
    print(class_names[:-1])
    print(f"\n 'Other' Class contains {len(other_files)} files.")
    print(
        f"   (Includes {len(skipped_top)} big fonts like {os.path.basename(skipped_top[0])})"
    )
    print(f"   (Includes {len(skipped_bottom)} small fonts)")

    # 2. Load Data Helper
    def load_pixels(files, label_idx):
        data_list = []
        labels_list = []
        for f in files:
            try:
                df = pd.read_csv(f)
                # Skip metadata columns
                pixels = df.iloc[:, 12:].values

                if pixels.shape[1] == 0:
                    continue

                data_list.append(pixels)
                labels_list.extend([label_idx] * len(pixels))
            except Exception as e:
                print(f"⚠️ Error reading {f}: {e}")
        return data_list, labels_list

    # 3. Process Files
    all_pixels = []
    all_labels = []

    # Load Selected Slice
    print("\n⏳ Processing Selected Slice...")
    for i, fname in enumerate(selected_files):
        print(f"   -> Loading {os.path.basename(fname)} (Class {i})...")
        p, l = load_pixels([fname], i)
        all_pixels.extend(p)
        all_labels.extend(l)

    # Load Others (The big ones + the small ones)
    print(" Processing 'Other' fonts ...")
    # The label for 'Other' is the length of selected_files (e.g., 10)
    other_label = len(selected_files)
    p, l = load_pixels(other_files, other_label)
    all_pixels.extend(p)
    all_labels.extend(l)

    # 4. Convert to Tensors
    print(" Converting to Tensors...")
    X_raw = np.vstack(all_pixels)
    y_raw = np.array(all_labels)

    # Reshape
    n_pixels = X_raw.shape[1]
    img_dim = int(np.sqrt(n_pixels))
    print(f"   Detected Image Size: {img_dim}x{img_dim}")

    X_tensor = torch.tensor(X_raw, dtype=torch.float32).reshape(-1, 1, img_dim, img_dim)
    y_tensor = torch.tensor(y_raw, dtype=torch.long)

    # 5. Save
    print(f" Saving to {OUTPUT_FILE}...")
    torch.save(
        {
            "X": X_tensor,
            "y": y_tensor,
            "class_names": class_names,
            "img_dim": (1, img_dim, img_dim),
        },
        OUTPUT_FILE,
    )
    print("Preprocessing Complete")


if __name__ == "__main__":
    preprocess_dataset()
