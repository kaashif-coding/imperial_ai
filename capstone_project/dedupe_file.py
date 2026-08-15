"""
One-time cleanup: remove duplicate (X, Y) rows from each function's
.npy files, caused by running append_weekly_data.py more than once
on the same week's data.

A row is considered a duplicate if BOTH its input (X) and output (Y)
exactly match an earlier row.
"""

import glob
import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, "initial_data")


def find_file(folder, keyword):
    matches = [
        f for f in glob.glob(os.path.join(folder, "*.npy"))
        if keyword in os.path.basename(f).lower()
    ]
    if not matches:
        raise FileNotFoundError(f"No file containing '{keyword}' found in {folder}")
    return matches[0]


print(f"Looking for function folders under: {BASE_DIR}\n")

for i in range(1, 9):
    folder = os.path.join(BASE_DIR, f"function_{i}")
    if not os.path.isdir(folder):
        print(f"⚠️  Function {i}: folder NOT FOUND at {folder} — skipping")
        continue

    inputs_path = find_file(folder, "input")
    outputs_path = find_file(folder, "output")

    X = np.load(inputs_path)
    Y = np.load(outputs_path)

    # Combine X and Y into one array so we can dedupe on the full (x, y) row
    combined = np.column_stack([X, Y])

    # np.unique sorts by default, so we instead dedupe while preserving order
    _, unique_idx = np.unique(combined, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)  # preserve original chronological order

    X_clean = X[unique_idx]
    Y_clean = Y[unique_idx]

    n_removed = X.shape[0] - X_clean.shape[0]

    if n_removed > 0:
        np.save(inputs_path, X_clean)
        np.save(outputs_path, Y_clean)
        print(f"Function {i}: {X.shape[0]} -> {X_clean.shape[0]} points "
              f"({n_removed} duplicate(s) removed)")
    else:
        print(f"Function {i}: {X.shape[0]} points, no duplicates found")

print("\nDone.")