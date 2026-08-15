"""
Capstone BBO challenge — propose the next query point for all 8 functions.

Expects a folder structure like:
    initial_data/
        function_1/  (contains an inputs .npy and an outputs .npy)
        function_2/
        ...
        function_8/

Update BASE_DIR below if your folder is somewhere else. The script
auto-detects the input/output files inside each function folder by
matching filenames containing "input" or "output".
"""

import glob
import os
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, "initial_data")
N_FUNCTIONS = 8

# --- Exploration/exploitation control, per function ---
# Round 3 change: ease off exploration for lower-dimensional functions
# (more data relative to their space, GP predictions more trustworthy now)
# while keeping exploration high for higher-dimensional functions (still
# very sparsely covered even with more total points).
BETA_BY_FUNCTION = {
    1: 2.0,   # 2D — easing back toward exploitation
    2: 2.0,   # 2D
    3: 2.5,   # 3D
    4: 3.0,   # 4D — keep exploring
    5: 3.0,   # 4D
    6: 3.5,   # 5D
    7: 3.5,   # 6D
    8: 4.0,   # 8D — still very sparse, push exploration hardest here
}


def find_file(folder, keyword):
    matches = [
        f for f in glob.glob(os.path.join(folder, "*.npy"))
        if keyword in os.path.basename(f).lower()
    ]
    if not matches:
        raise FileNotFoundError(f"No file containing '{keyword}' found in {folder}")
    return matches[0]


def propose_next_point(X, Y, beta, seed=0):
    n_points, n_dims = X.shape

    # Wider length-scale bounds let the GP consider shorter length-scales
    # (i.e. less smoothing), which keeps posterior uncertainty higher away
    # from observed points rather than assuming broad similarity too readily.
    kernel = RBF(length_scale=0.2, length_scale_bounds=(1e-3, 1e1)) + WhiteKernel(
        noise_level=1e-6, noise_level_bounds=(1e-10, 1e-1)
    )
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=8)
    gp.fit(X, Y)

    np.random.seed(seed)
    if n_dims <= 3:
        # Denser grid = more thorough exploration coverage per dimension
        grid_res = 35 if n_dims == 3 else 100
        axes = [np.linspace(0, 1, grid_res) for _ in range(n_dims)]
        mesh = np.meshgrid(*axes)
        candidates = np.column_stack([m.ravel() for m in mesh])
    else:
        # More random candidates = better chance of finding genuinely
        # under-explored regions in higher dimensions
        n_candidates = 50000
        candidates = np.random.uniform(0, 1, size=(n_candidates, n_dims))

    post_mean, post_std = gp.predict(candidates, return_std=True)
    acquisition = post_mean + beta * post_std

    best_idx = np.argmax(acquisition)
    next_x = candidates[best_idx]
    return next_x, n_points, n_dims


results = {}

for i in range(1, N_FUNCTIONS + 1):
    folder = os.path.join(BASE_DIR, f"function_{i}")
    if not os.path.isdir(folder):
        print(f"Function {i}: folder not found ({folder}), skipping")
        continue

    inputs_path = find_file(folder, "input")
    outputs_path = find_file(folder, "output")

    X = np.load(inputs_path)
    Y = np.load(outputs_path)

    beta = BETA_BY_FUNCTION[i]
    next_x, n_points, n_dims = propose_next_point(X, Y, beta)
    formatted = "-".join(f"{v:.6f}" for v in next_x)
    results[i] = formatted

    print(f"Function {i} ({n_dims}-D, {n_points} points so far, beta={beta}):")
    print(f"  {formatted}\n")

print("=" * 50)
print("Summary — copy these into the portal:")
print("=" * 50)
for i, formatted in results.items():
    print(f"Function {i}: {formatted}")