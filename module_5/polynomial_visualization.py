import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Generate data
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 3 * X.squeeze()**3 - 2 * X.squeeze()**2 + X.squeeze() + 5 + np.random.randn(100) * 100

# =============================================================================
# PART 1: ORIGINAL DATA
# =============================================================================
print("="*70)
print("PART 1: ORIGINAL DATA")
print("="*70)
print("\nFirst 5 rows of original data:")
print("X values        y values")
for i in range(5):
    print(f"{X[i,0]:8.3f}  →  {y[i]:10.2f}")

plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.6, s=30)
plt.xlabel('X', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('Original Data: Just (X, y) points\nModel needs to find the relationship!', fontsize=14)
plt.grid(True, alpha=0.3)
plt.savefig('part1_original_data.png', dpi=100, bbox_inches='tight')
print("\n✓ Saved: part1_original_data.png")
plt.close()

# =============================================================================
# PART 2: POLYNOMIAL TRANSFORMATION
# =============================================================================
print("\n" + "="*70)
print("PART 2: POLYNOMIAL TRANSFORMATION")
print("="*70)

# Show transformation for a single example
example_x = np.array([[5.0]])
print(f"\nExample: X = {example_x[0,0]}")
print("\nTransformations for different degrees:")

for deg in [1, 2, 3]:
    poly = PolynomialFeatures(degree=deg)
    x_transformed = poly.fit_transform(example_x)
    feature_names = poly.get_feature_names_out(['x'])

    print(f"\nDegree {deg}:")
    print(f"  Features: {list(feature_names)}")
    print(f"  Values:   {x_transformed[0]}")

# Visualize transformation for first 5 data points
print("\n" + "-"*70)
print("First 5 rows transformed with degree=3:")
print("-"*70)
poly3 = PolynomialFeatures(degree=3)
X_poly3 = poly3.fit_transform(X[:5])
print("Original X  →  [1,        x,        x²,       x³]")
for i in range(5):
    print(f"{X[i,0]:8.3f}  →  {X_poly3[i]}")

# =============================================================================
# PART 3: MODEL FITTING - COMPARING DIFFERENT DEGREES
# =============================================================================
print("\n" + "="*70)
print("PART 3: MODEL FITTING")
print("="*70)
print("\nTrue formula: y = 5 + 1*x - 2*x² + 3*x³ (+ noise)")
print("\nWhat each model learns:\n")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

for i, degree in enumerate([1, 2, 3, 6]):
    # Transform the data
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    # Fit the model
    model = LinearRegression()
    model.fit(X_poly, y)

    # Print learned coefficients
    feature_names = poly.get_feature_names_out(['x'])
    print(f"Degree {degree}:")
    print(f"  Intercept (β₀): {model.intercept_:.2f}")
    for j, (name, coef) in enumerate(zip(feature_names[1:], model.coef_[1:])):
        print(f"  {name:6s} (β_{j+1}): {coef:.2f}")

    # Create smooth prediction line
    X_plot = np.linspace(0, 10, 300).reshape(-1, 1)
    X_plot_poly = poly.transform(X_plot)
    y_plot = model.predict(X_plot_poly)

    # Plot
    ax = axes[i]
    ax.scatter(X, y, alpha=0.4, s=20, label='Data points')
    ax.plot(X_plot, y_plot, 'r-', linewidth=3, label=f'Model fit (degree {degree})')

    # Add true function for comparison
    y_true = 3 * X_plot.squeeze()**3 - 2 * X_plot.squeeze()**2 + X_plot.squeeze() + 5
    ax.plot(X_plot, y_true, 'g--', linewidth=2, alpha=0.7, label='True function (no noise)')

    ax.set_xlabel('X', fontsize=11)
    ax.set_ylabel('y', fontsize=11)
    ax.set_title(f'Degree {degree} Polynomial', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Add text showing if it's good fit
    if degree == 1:
        ax.text(0.5, 0.95, 'UNDERFITTING\n(too simple)',
                transform=ax.transAxes, fontsize=11, color='red',
                ha='center', va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    elif degree == 3:
        ax.text(0.5, 0.95, 'GOOD FIT\n(matches true degree)',
                transform=ax.transAxes, fontsize=11, color='green',
                ha='center', va='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    elif degree == 6:
        ax.text(0.5, 0.95, 'OVERFITTING\n(too wiggly)',
                transform=ax.transAxes, fontsize=11, color='red',
                ha='center', va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    print()

plt.tight_layout()
plt.savefig('part3_model_fitting.png', dpi=100, bbox_inches='tight')
print("✓ Saved: part3_model_fitting.png")
plt.close()

# =============================================================================
# PART 4: HOW THE MODEL "DERIVES" THE POLYNOMIAL
# =============================================================================
print("\n" + "="*70)
print("PART 4: HOW THE MODEL DERIVES THE POLYNOMIAL")
print("="*70)
print("\nThe model uses LINEAR REGRESSION on the transformed features:")
print("\n1. We transform X into polynomial features")
print("2. LinearRegression finds coefficients (β) that minimize error")
print("3. It solves: minimize Σ(y - (β₀ + β₁*x + β₂*x² + β₃*x³))²")
print("\nCompare learned vs true coefficients (for degree 3):")
print("-"*70)

poly3 = PolynomialFeatures(degree=3)
X_poly3 = poly3.fit_transform(X)
model3 = LinearRegression()
model3.fit(X_poly3, y)

comparison = {
    'Constant': (5, model3.intercept_),
    'x¹':       (1, model3.coef_[1]),
    'x²':       (-2, model3.coef_[2]),
    'x³':       (3, model3.coef_[3])
}

print(f"{'Term':<10} {'True Value':<15} {'Learned Value':<15} {'Difference'}")
print("-"*70)
for term, (true_val, learned_val) in comparison.items():
    diff = abs(true_val - learned_val)
    print(f"{term:<10} {true_val:<15.2f} {learned_val:<15.2f} {diff:.2f}")

print("\nNote: Differences are due to the random noise we added!")
print("The model successfully 'discovered' the polynomial relationship!")
print("\n" + "="*70)
print("✓ Analysis complete! Check the saved images.")
print("="*70)

