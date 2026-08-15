import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

try:
    from IPython.display import clear_output
except Exception:
    def clear_output(wait=True):
        pass


# Reproducible random seed so you can compare runs
np.random.seed(0)

# Parameters of the problem
real_noise_std = 1e-10
noise_assumption = 1e-10
rbf_lengthscale = 0.1
beta = 1.96

# Draw random function parameters
modes = np.random.randint(1, 5)
std = np.random.uniform(low=0.005, high=0.05, size=modes)
means = np.random.uniform(size=modes)
amps = np.random.uniform(size=modes) * (2 - 1) + 1


def calc_function(x):
    exp = -(x - means) ** 2 / std
    y = amps * np.exp(exp)
    return np.sum(y)


# Define the kernel of the GP
kernel = RBF(length_scale=rbf_lengthscale, length_scale_bounds='fixed')
model = GaussianProcessRegressor(kernel=kernel, alpha=noise_assumption)

# Initialise query lists and maximum observations
X, Y = [], []
max_obs = 0

# Initialise grid for plots
x_grid = np.linspace(0, 1, 101).reshape(-1, 1)

# Number of queries in the optimisation loop
num_queries = 5

for i in range(0, num_queries):
    clear_output(wait=True)
    model = GaussianProcessRegressor(kernel=kernel)

    if i != 0:
        model.fit(np.array(X).reshape(-1, 1), np.array(Y).reshape(-1, 1))

    # Calculate posterior mean and std
    post_mean, post_std = model.predict(x_grid, return_std=True)
    post_mean, post_std = post_mean.squeeze(), post_std.squeeze()

    # Plot the current GP posterior
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(max_obs + 1, 3))
    ax.set_ylabel('f(x)')
    ax.set_xlabel('x')
    ax.set_title(f'So far you have selected {i} queries, you have {10 - i} left.')
    ax.scatter(X, Y, c='r', marker='x', s=100)
    ax.plot(x_grid.squeeze(), post_mean, label='GP Posterior Mean')
    ax.fill_between(
        x_grid.squeeze(),
        post_mean - beta * post_std,
        post_mean + beta * post_std,
        alpha=0.2,
        label=f'{beta} Standard Deviations',
    )
    ax.legend()
    plt.show()

    # Pause so you can inspect the current state in the notebook-style loop
    #input('Press Enter to continue to the next chosen point...')

    # Variance-based acquisition: pick the point with the highest uncertainty
    acquisition_function = post_std ** 2

    # Debugger breakpoint right where the acquisition function is computed
    if i == 0:
        print('First iteration: using a random starting point.')
        x = np.random.uniform(0, 1)
    else:
        grid = x_grid.squeeze()
        x = grid[np.argmax(acquisition_function)]

    # Append data, calculate function, and sort by observation value
    X.append(x)
    y = calc_function(x) + np.random.normal(scale=real_noise_std)
    Y.append(y)
    X = [x for _, x in sorted(zip(Y, X), reverse=True)]
    Y.sort(reverse=True)
    max_obs = max(max_obs, y)

clear_output()

# Calculate the real function on a dense grid
x_grid = np.linspace(0, 1, 1001)
y_real = []
best_obs_grid = 0
for x in x_grid:
    y = calc_function(x)
    y_real.append(y)
    best_obs_grid = max(best_obs_grid, y)

# Final GP posterior
model.fit(np.array(X).reshape(-1, 1), np.array(Y).reshape(-1, 1))
post_mean, post_std = model.predict(x_grid.reshape(-1, 1), return_std=True)
post_mean, post_std = post_mean.squeeze(), post_std.squeeze()

# Final plot
fig, ax = plt.subplots(figsize=(15, 7))
ax.plot(x_grid, y_real, 'k', label='f(x)')
ax.scatter(X, Y, c='r', marker='x', label='Queries', s=100)
ax.set_ylabel('f(x)')
ax.set_xlabel('x')
ax.set_xlim(0, 1)
ax.set_ylim(bottom=0)
ax.set_title('Real function and all queries')
ax.plot(x_grid.squeeze(), post_mean, label='GP Posterior Mean')
ax.fill_between(
    x_grid.squeeze(),
    post_mean - beta * post_std,
    post_mean + beta * post_std,
    alpha=0.2,
    label=f'{beta} Standard Deviations',
)
ax.legend()
plt.show()

print('Maximum (by Grid-Search):')
print(best_obs_grid)
print('Best Observation (by Variance-based acquisition):')
print(max_obs)
