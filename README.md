# imperial_ai
# Black-Box Optimisation Capstone — Stage 2

## Overview

This repository documents my work on the **Black-Box Optimisation (BBO) challenge**, part of Imperial College Business School's Executive Programme in Machine Learning and AI. The goal is to find the maximum of eight unknown ("black-box") functions of increasing dimensionality, using **Bayesian optimisation** to select each new query point efficiently — without ever seeing the functions' underlying equations.

Each week, I submit one new input point per function to a course portal. The portal evaluates the true (hidden) function at that point and returns the result, which is added to my growing dataset and used to inform the next query. This README documents Stage 2 of the project: my third round of queries, and the reasoning behind the modelling choices made so far.

## Purpose

Real-world optimisation problems often involve functions that are expensive, slow, or impossible to evaluate exhaustively — tuning a machine learning model's hyperparameters, calibrating a financial pricing model, or optimising a physical experiment, for example. In these settings, you can't run a full grid search; every evaluation costs time, money, or computational resources.

This project simulates that exact constraint: each of the 8 functions can only be queried once per week, so the objective isn't just to find good inputs — it's to find them **efficiently**, using a principled strategy to decide where to look next given everything learned so far.

## The eight functions

| Function | Dimensions | Represents |
|---|---|---|
| 1 | 2D | Detecting contamination sources from proximity readings |
| 2 | 2D | Maximising a noisy log-likelihood score |
| 3 | 3D | Minimising side effects across three drug compounds (framed as maximisation) |
| 4 | 4D | Warehouse placement optimisation via ML-approximated hyperparameters |
| 5 | 4D | Maximising yield in a unimodal chemical process |
| 6 | 5D | Optimising a cake recipe across five ingredients |
| 7 | 6D | Tuning six ML hyperparameters for model performance |
| 8 | 8D | Tuning eight ML hyperparameters for validation accuracy |

Each function takes an n-dimensional input (values between 0 and 1) and returns a single scalar output to be maximised. All functions started with a baseline dataset that scales with dimensionality — lower-dimensional functions started with fewer points, higher-dimensional functions with more, since higher-dimensional spaces need more initial coverage to be usable at all.

## Inputs and outputs

**Input:** for each function, a query point `x1-x2-...-xn`, where each value is between 0 and 1, specified to six decimal places, and `n` matches that function's dimensionality (2 to 8).

**Output:** a single scalar score returned by the portal after processing, representing the (noisy) value of the true function at the submitted point.

**Data storage:** each function's accumulated history is stored as a pair of NumPy arrays — `inputs.npy` (shape: `n_points × n_dims`) and `outputs.npy` (shape: `n_points`) — with row `i` of the inputs corresponding to element `i` of the outputs.

## Technical approach

### 1. Surrogate modelling with Gaussian Processes

Since each function is expensive to evaluate (one query per week), I use a **Gaussian Process (GP)** as a surrogate model — a cheap, probabilistic stand-in for the true function. Given the points evaluated so far, the GP produces two things at any candidate input: a predicted value (posterior mean) and an uncertainty estimate (posterior standard deviation). I use `scikit-learn`'s `GaussianProcessRegressor` with a combined RBF + WhiteKernel, which lets the model account for both smooth structure in the function and observation noise.

### 2. Acquisition function: Upper Confidence Bound (UCB)

Rather than trusting the GP's prediction blindly, I select the next query point using an **acquisition function** — a rule that scores every candidate point by combining the GP's prediction and its uncertainty:

```
UCB(x) = predicted_mean(x) + β × predicted_std(x)
```

The `β` parameter controls the exploration/exploitation trade-off: a higher β favours points the model is uncertain about (exploration); a lower β favours points the model already predicts will score well (exploitation).

### 3. Candidate search strategy

For lower-dimensional functions (≤3D), I evaluate the acquisition function over a dense grid covering the input space. For higher-dimensional functions, a dense grid becomes computationally infeasible (grid size grows exponentially with dimensionality — the "curse of dimensionality"), so I instead sample tens of thousands of random candidate points and select the best-scoring one.

### 4. Adapting strategy round by round

My strategy has evolved as the dataset has grown:

- **Round 1:** neutral UCB (β≈2), no prior signal to lean on.
- **Round 2:** shifted heavily toward exploration (higher β, wider kernel bounds), since early data was too sparse to trust the model's predictions.
- **Round 3:** moved to a **per-function β**, easing exploration for lower-dimensional functions (where the GP's predictions are starting to stabilise) while keeping exploration high for higher-dimensional functions, which remain sparsely covered even as the total point count grows.

### 5. Connecting to regression and SVMs

This module's coursework on linear/logistic regression and SVMs prompted a critical look at whether simpler models could apply here:

- **Linear/logistic regression** would violate core assumptions in this setting — the response surfaces are non-linear and often multi-modal, and with as few as 12 points across up to 8 dimensions, there isn't enough data per parameter to fit a stable regression. That said, regression's interpretability (a clear coefficient per input) is something the GP-based approach lacks.
- **Support Vector Machines**, particularly a soft-margin, kernel (RBF) SVM, offer a promising alternative framing: rather than predicting an exact output value, a classifier could distinguish "high-performing" from "low-performing" regions of the input space, offering a faster, more interpretable way to triage where to search before running the full GP + UCB pipeline — especially valuable as dimensionality (and candidate sparsity) increases.

## Repository structure

```
capstone_project/
├── initial_data/
│   ├── function_1/
│   │   ├── inputs.npy
│   │   └── outputs.npy
│   ├── function_2/
│   ...
│   └── function_8/
├── append_weekly_data.py     # appends each week's new (input, output) pair per function
├── dedupe_function_data.py   # one-time cleanup for accidental duplicate rows
├── compute_all_queries.py    # fits GP + runs UCB acquisition, proposes next query per function
└── README.md
```

## How to reproduce

1. Ensure `initial_data/function_1` through `function_8` each contain an `inputs.npy` and `outputs.npy` file.
2. After receiving each week's result, update `append_weekly_data.py` with the new (input, output) pair and run it to append the new data point.
3. Run `compute_all_queries.py` to fit a GP to the current dataset for each function and print the next recommended query point, formatted for direct submission to the course portal.
4. Submit the printed query points and repeat weekly as new data arrives.

## Requirements

```
numpy
scikit-learn
```

## Notes on interpretability and limitations

As the dataset grows, the main limitation remains **dimensionality-dependent sparsity**: even as point counts increase, higher-dimensional functions (6D, 8D) remain far less thoroughly covered relative to their space than the lower-dimensional functions. No individual dimension has yet emerged as clearly irrelevant, though examining the GP's fitted kernel length-scales per dimension is a natural next step — a very large fitted length-scale for a given input would suggest the output is relatively insensitive to that dimension.
