import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

real_noise_std = 1e-10
noise_assumption = 1e-10

rbf_lengthscale = 0.1

#Acquisition function parameter
beta = 1.96

#Draw random function parameters
modes = np.random.randint(1, 5)
std = np.random.uniform(low = 0.005, high = 0.05, size = modes)
means = np.random.uniform(size = modes)
amps = np.random.uniform(size = modes) * (2 - 1) + 1

def calc_function(x):
  exp = -(x - means) ** 2 / std
  y = amps * np.exp(exp)
  return np.sum(y)



xs = np.linspace(-10, 10, 500)
ys = np.array([calc_function(x) for x in xs])

plt.plot(xs, ys)
plt.xlabel("x")
plt.ylabel("calc_function(x)")
plt.title("Output of calc_function")
plt.grid(True)
plt.show()