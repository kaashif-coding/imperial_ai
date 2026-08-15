# Black-Box Optimisation (BBO) Capstone Project

## Section 1: Project overview

This project tackles a **black-box optimisation (BBO)** challenge: finding the maximum of eight unknown functions, each of increasing input dimensionality (from 2D up to 8D), without ever seeing their underlying equations. Each function can only be queried once per week, so the challenge isn't just to find a good result eventually - it's to find one **efficiently**, using a principled strategy to decide where to look next given everything learned so far.

The high-level idea is one that comes up constantly in real-world machine learning and quantitative work: you often have a function you want to optimise (a model's validation accuracy, a pricing model's calibration error, a physical process's yield) but each evaluation is slow, expensive, or resource-constrained, so you can't brute-force a full grid search. Bayesian optimisation addresses exactly this by building a probabilistic model of the function from the data collected so far, and using that model to intelligently choose the next point to test - balancing the trade-off between exploring uncertain regions and exploiting regions already known to perform well.

This project is directly relevant to my current and future career. I work in pricing and optimisation technology in prime finance, where calibrating models against real, costly-to-obtain market and desk outcomes is structurally the same problem: I can't exhaustively test every parameter combination, and each real evaluation has a genuine cost. Beyond the immediate technical skills (Gaussian Processes, acquisition functions, candidate search strategies), the discipline this project builds - explicitly separating "what does my model currently believe" from "what should I do about that belief" - is a mental model I expect to carry directly into how I approach model calibration and experimentation more broadly, including as I look to move into more technical, trading-aligned roles.

## Section 2: Inputs and outputs

**Input:** for each function, a query point in the format `x1-x2-x3-...-xn`, where:
- Each value `xi` is a number between 0 and 1
- Each value is specified to exactly six decimal places
- `n` matches that function's input dimensionality (ranging from 2 to 8 across the eight functions)

Example, for a 2D function: `0.593220-0.881356`
Example, for a 4D function: `0.406863-0.467406-0.429753-0.452384`

**Output:** a single scalar score returned after the query is processed, representing the (noisy) value of the true, hidden function evaluated at the submitted input. This is stored alongside the corresponding input as one row of accumulated data per function.

**Data storage:** each function's full history of (input, output) pairs is stored as a pair of NumPy arrays - `inputs.npy` (shape: `n_points × n_dims`) and `outputs.npy` (shape: `n_points`) - with row `i` of the inputs always corresponding to element `i` of the outputs.

## Section 3: Challenge objectives

The objective across all eight functions is **maximisation** - for each function, the goal is to find the input that produces the highest possible output score. (Some functions represent real-world problems that are naturally framed as minimisation, such as minimising side effects in a drug discovery scenario or minimising negative-scored recipe attributes, but these are transformed into maximisation problems, e.g. by taking the negative of the original quantity.)

The key constraints shaping the strategy are:

- **Query budget:** only one new query per function is allowed each week, across 13 rounds total - there's no room to "waste" a query on a clearly uninformative point.
- **Response delay:** results aren't immediate - each week's query is processed and returned the following week, meaning strategy has to be decided with only the data available at that point, not adjusted mid-week based on partial feedback.
- **Unknown function structure:** the functions' true forms are completely hidden. I know only high-level context (e.g. "this represents a chemical yield process, typically unimodal" or "this is a multi-modal, noisy log-likelihood surface"), which informs modelling choices (like kernel smoothness assumptions) but not the actual shape.
- **Increasing dimensionality:** as dimensionality grows (from 2D up to 8D), the same number of data points covers a proportionally tiny fraction of the space (the "curse of dimensionality"), which fundamentally limits how confident any model can be in the higher-dimensional functions this early in the process.

## Section 4: Technical approach

*(This section is a living record, updated as my approach evolves each round.)*

### Core method: Gaussian Process surrogate + UCB acquisition

My core approach uses a **Gaussian Process (GP)** as a surrogate model for each unknown function. Given the points evaluated so far, the GP provides both a predicted value (posterior mean) and an uncertainty estimate (posterior standard deviation) at any candidate input - implemented via `scikit-learn`'s `GaussianProcessRegressor`, using a combined RBF + WhiteKernel to capture both smooth structure and observation noise.

To choose each week's query point, I use the **Upper Confidence Bound (UCB)** acquisition function:

```
UCB(x) = predicted_mean(x) + β × predicted_std(x)
```

where `β` controls the exploration/exploitation balance - higher values favour uncertain regions (exploration), lower values favour regions already predicted to score well (exploitation). For lower-dimensional functions I search a dense grid of candidate points; for higher-dimensional functions, a dense grid becomes computationally infeasible, so I instead sample tens of thousands of random candidates and select the best-scoring one.

### How this has evolved across my first three rounds

- **Round 1:** a neutral starting point - standard UCB, no bias toward exploration or exploitation, since I had no prior signal from my own submissions yet.
- **Round 2:** with only 10-11 points per function, I judged the GP's predictions too unreliable to trust, so I deliberately shifted toward exploration - raising β and widening the kernel's length-scale bounds - to prioritise mapping out each function more broadly before narrowing in.
- **Round 3:** moved to a **per-function β** rather than one fixed value. For lower-dimensional functions, where the posterior mean is starting to look more stable between rounds, I eased β back down toward exploitation. For higher-dimensional functions, which remain sparsely covered even as total point counts grow, I kept β high to continue prioritising exploration there.

### Where SVMs, regression, and other techniques fit in

Reflecting on the broader ML toolkit covered in this programme:

- **Linear/logistic regression** would struggle here - the response surfaces are clearly non-linear and often multi-modal, and with as few as a dozen points across up to 8 dimensions, there isn't enough data per parameter to fit a stable regression model. That said, regression's interpretability (a direct coefficient per input) is something my current GP-based approach lacks, and is a trade-off worth keeping in mind.
- **Support Vector Machines**, specifically a soft-margin, kernel (RBF) SVM, are a promising direction I'm considering for future rounds: rather than predicting an exact output value, a classifier could distinguish "high-performing" from "low-performing" regions of the input space. This could offer a faster, more interpretable way to triage which regions are worth exploring before running the full GP + UCB pipeline - particularly valuable as dimensionality increases and the candidate space becomes sparser relative to available data.

### What makes this approach thoughtful (and where it's still limited)

The main thing I'd highlight is treating exploration/exploitation balance as a **per-function, evolving decision** rather than a fixed setting applied uniformly - since each function's dimensionality and data coverage genuinely differ, applying the same β everywhere would ignore information I actually have available.

The clearest current limitation is dimensionality-dependent sparsity: even as total data grows, higher-dimensional functions remain far less thoroughly covered relative to their space than lower-dimensional ones, which keeps the acquisition function close to "explore broadly" almost regardless of β in those cases. I also haven't yet identified any input dimension as clearly irrelevant - a natural next step would be examining the GP's fitted per-dimension length-scales, where a very large fitted length-scale would suggest low sensitivity to that particular input.