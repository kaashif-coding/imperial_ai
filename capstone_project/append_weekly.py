"""
Append this week's new (input, output) pair to each function's existing
.npy files.

Each week: paste the new values from the email into new_inputs / new_outputs
below (in Function 1 -> 8 order), then run this script before
compute_all_queries.py.

Expects:
    initial_data/
        function_1/  (input .npy, output .npy)
        function_2/
        ...
        function_8/
"""

import glob
import os
import numpy as np

# Anchor BASE_DIR to this script's own folder, so it works regardless of
# what directory PyCharm/the terminal happens to be running from.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, "initial_data")

# --- Paste this week's new data here, in Function 1 -> 8 order ---
new_inputs = [
    np.array([0.762712, 0.796610]),
    np.array([0.677966, 0.898305]),
    np.array([0.333333, 0.000000, 0.541667]),
    np.array([0.436013, 0.416305, 0.315606, 0.431708]),
    np.array([0.423915, 0.768388, 0.996361, 0.983487]),
    np.array([0.234512, 0.491969, 0.122340, 0.820606, 0.038577]),
    np.array([0.135011, 0.352687, 0.311913, 0.070196, 0.207284, 0.756211]),
    np.array([0.065376, 0.735412, 0.004793, 0.171499, 0.835374, 0.459132, 0.034125, 0.717182]),
]

new_outputs = [
    8.025234978553625e-33,
    0.5184746989499363,
    -0.11609163559219807,
    -0.2576031170180175,
    2349.884984342567,
    -1.093519510160005,
    1.3739239975855386,
    9.5487045152316,
]


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

    new_x = new_inputs[i - 1]
    new_y = new_outputs[i - 1]

    assert new_x.shape[0] == X.shape[1], (
        f"Function {i}: new point has {new_x.shape[0]} dims, "
        f"expected {X.shape[1]}"
    )

    # Guard against accidentally re-running this script on the same week's
    # data — skip if this exact (x, y) pair is already the last row.
    if X.shape[0] > 0 and np.allclose(X[-1], new_x) and np.isclose(Y[-1], new_y):
        print(f"Function {i}: last point already matches new data — skipping "
              f"(already appended, {X.shape[0]} points)")
        continue

    X_updated = np.vstack([X, new_x])
    Y_updated = np.append(Y, new_y)

    np.save(inputs_path, X_updated)
    np.save(outputs_path, Y_updated)

    print(f"Function {i}: {X.shape[0]} -> {X_updated.shape[0]} points "
          f"(added {new_x}, {new_y})")

print("\nDone. Now re-run compute_all_queries.py to get this week's next query points.")