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
    np.array([0.593220, 0.881356]),
    np.array([0.813559, 0.966102]),
    np.array([0.416667, 0.333333, 0.541667]),
    np.array([0.406863, 0.467406, 0.429753, 0.452384]),
    np.array([0.299819, 0.810892, 0.999108, 0.994635]),
    np.array([0.344861, 0.256363, 0.496530, 0.776213, 0.077734]),
    np.array([0.042079, 0.442993, 0.252227, 0.192031, 0.243311, 0.745455]),
    np.array([0.000441, 0.141408, 0.002014, 0.027526, 0.855212, 0.381535, 0.075849, 0.420913]),
]

new_outputs = [
    -1.2752566952547017e-46,
    0.010329637558293425,
    -0.03584366615693301,
    -0.8402833524101285,
    2642.3386449207323,
    -0.38705999892677373,
    1.633862741346932,
    9.8663687636541,
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