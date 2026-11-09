# %%
#author: Noureldin Elnems  , noe2466@thi.de , Group: 7 

# ============================================================
# 4.1.a Generate the synthetic dataset
# ============================================================

import numpy as np

# We fix the random seed so the generated dataset stays the same
np.random.seed(42)
M = 20

# coefficients of the quadratic function:
a = 0.25
b = -5
c = 0.2

# Generate M random x-values uniformly from the interval [0, 30].
x = np.random.uniform(0, 30, M)
x = np.sort(x)
noise = np.random.normal(loc=0, scale=4.5, size=M)

# Compute the target values y using the required formula:
y = a * x**2 + b * x + c + noise

# reshaping x values, the input data X must usually be a 2D array
X = x.reshape(-1, 1)

# Print basic information so we can verify the generated dataset.
print("Dataset generated successfully.")
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# Show the first 5 values to inspect the data quickly.
print("x-values:")
print(x)
print("Noise values:")
print(noise)
print("y-values:")
print(y)

# ============================================================
# 4.1.b Build three regression models with different complexity
# ============================================================
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Model 1: Linear Regression
model_lin = LinearRegression()
model_lin.fit(X, y)


# Model 2: Polynomial Regression of degree 2
model_quad = make_pipeline(
    PolynomialFeatures(degree=2),
    LinearRegression()
)

# Train the quadratic model on the same dataset.
model_quad.fit(X, y)

# Model 3: Polynomial Regression of degree 10
model_deg10 = make_pipeline(
    PolynomialFeatures(degree=10),
    LinearRegression()
)

# Train the degree-10 model on the dataset.
model_deg10.fit(X, y)


# Prepare a dense set of x-values for plotting the fitted curves
x_plot = np.linspace(0, 30, 5000)
X_plot = x_plot.reshape(-1, 1)

# Predict the output of each model on the dense x-range
y_lin = model_lin.predict(X_plot)
y_quad = model_quad.predict(X_plot)
y_deg10 = model_deg10.predict(X_plot)

# Print confirmation and basic information
print("\nAll three models were trained successfully.")
print("\nModel 1: Linear Regression")
print("Model 2: Polynomial Regression (degree 2)")
print("Model 3: Polynomial Regression (degree 10)")
print("\nPrediction arrays shapes:")
print("y_lin shape:", y_lin.shape)
print("y_quad shape:", y_quad.shape)
print("y_deg10 shape:", y_deg10.shape)

# ============================================================
# 4.1.c Plot data and models (final improved version)
# ============================================================

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

# Scatter plot of original data
plt.scatter(x, y, label="Generated data")

# Model curves
plt.plot(x_plot, y_lin, label="Linear model (degree 1)")
plt.plot(x_plot, y_quad, label="Quadratic model (degree 2)")
plt.plot(x_plot, y_deg10, label="Polynomial model (degree 10)")

# Main title
plt.title("Regression Models Comparison")

# Subtitle (extra clarity for the assignment)
plt.suptitle("Demonstration of Underfitting and Overfitting", fontsize=10)

# Axis labels (this is what “multiple axes” refers to)
plt.xlabel("x values")
plt.ylabel("y values")

# Legend and grid
plt.legend()
plt.grid(True)
plt.show()

# ============================================================
# 4.1.d Compute the mean squared error (MSE) for each model
# ============================================================

# Import the MSE function only now, because this part is about evaluation.
from sklearn.metrics import mean_squared_error

# Compute predictions of each model on the ORIGINAL dataset
y_lin_train = model_lin.predict(X)
y_quad_train = model_quad.predict(X)
y_deg10_train = model_deg10.predict(X)

# Compute the Mean Squared Error for each model
mse_lin = mean_squared_error(y, y_lin_train)
mse_quad = mean_squared_error(y, y_quad_train)
mse_deg10 = mean_squared_error(y, y_deg10_train)

# Print the results clearly
print("Mean Squared Error (MSE) of each model on the generated dataset:\n")
print(f"Linear model (degree 1): {mse_lin:.4f}")
print(f"Quadratic model (degree 2): {mse_quad:.4f}")
print(f"Polynomial model (degree 10): {mse_deg10:.4f}")


