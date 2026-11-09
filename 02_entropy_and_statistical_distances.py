# %%
# =========================================================
# Required libraries
# =========================================================
# pandas: used to read CSV files and work with tabular data
import pandas as pd

# numpy: used for numerical operations, arrays, logarithms,
# matrix operations, covariance, correlation, etc.
import numpy as np

# matplotlib.pyplot: used for plotting figures, scatter plots, titles, labels, etc.
import matplotlib.pyplot as plt

# Ellipse from matplotlib.patches: used to draw covariance ellipses
from matplotlib.patches import Ellipse


# =========================================================
# 2.1.a
# Load the binary entropy dataset and plot X and Y with their labels
# =========================================================

# pd.read_csv(...) reads a CSV file and returns a pandas DataFrame
# A DataFrame is like a table with rows and columns
df = pd.read_csv("data_entropy_binary.csv")

# df.head() prints the first 5 rows of the table
print(df.head())

# df.columns prints the names of the columns
print(df.columns)

# Define the feature columns and label columns for the two random variables
x_cols = ["x_1", "x_2"]
x_label_col = "x_label"

y_cols = ["y_1", "y_2"]
y_label_col = "y_label"

# plt.subplots(1, 2, figsize=(12, 5)) creates one figure with 2 subplots
# axes[0] is the left plot, axes[1] is the right plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# df[column].unique() returns the distinct values الموجودة في العمود
# sorted(...) sorts them in ascending order
for label in sorted(df[x_label_col].unique()):
    # df[df[x_label_col] == label] filters rows where the label has a specific value
    subset = df[df[x_label_col] == label]

    # scatter(x, y) plots points using x-axis and y-axis values
    axes[0].scatter(subset[x_cols[0]], subset[x_cols[1]], label=label, s=20)

axes[0].set_title("Random Variable X")
axes[0].set_xlabel("x_1")
axes[0].set_ylabel("x_2")
axes[0].legend()

for label in sorted(df[y_label_col].unique()):
    subset = df[df[y_label_col] == label]
    axes[1].scatter(subset[y_cols[0]], subset[y_cols[1]], label=label, s=20)

axes[1].set_title("Random Variable Y")
axes[1].set_xlabel("y_1")
axes[1].set_ylabel("y_2")
axes[1].legend()

# tight_layout() improves spacing between subplots
plt.tight_layout()

# show() displays the figure
plt.show()


# =========================================================
# 2.1.b
# Compute the probabilities of each label
# =========================================================

def compute_probabilities(labels):
    """
    This function computes the probability of each label.

    Input:
        labels -> a pandas Series containing labels such as 0, 1, 2, ...

    Output:
        probabilities -> dictionary in the form:
                         {label: probability}
    """
    probabilities = {}

    # len(labels) gives the total number of label values
    total_count = len(labels)

    # labels.unique() returns the distinct labels only
    for label in labels.unique():
        # (labels == label) creates a boolean Series: True where the label matches
        # .sum() counts how many True values exist
        label_count = (labels == label).sum()

        # probability = count of label / total number of labels
        probabilities[label] = float(label_count / total_count)

    return probabilities


x_probs = compute_probabilities(df["x_label"])
y_probs = compute_probabilities(df["y_label"])

print("Probabilities for X:", x_probs)
print("Probabilities for Y:", y_probs)


# =========================================================
# 2.1.c
# Compute entropy
# =========================================================

def compute_entropy(probabilities):
    """
    This function computes entropy using the formula:

        H(X) = - sum( p * log2(p) )

    Input:
        probabilities -> dictionary of probabilities

    Output:
        entropy -> one numeric value
    """
    entropy = 0

    # probabilities.values() returns only the probability values from the dictionary
    for p in probabilities.values():
        # We check p > 0 because log(0) is undefined
        if p > 0:
            # np.log2(p) computes logarithm base 2
            entropy -= p * np.log2(p)

    return entropy


# =========================================================
# 2.1.d
# Print entropy and maximal entropy for the binary dataset
# =========================================================

def compute_max_entropy(labels):
    """
    This function computes the maximal entropy.

    Formula:
        H_max = log2(number_of_classes)

    Input:
        labels -> pandas Series of labels

    Output:
        maximum possible entropy for this number of classes
    """
    # nunique() returns the number of distinct labels in the column
    num_classes = labels.nunique()

    # np.log2(...) calculates log base 2
    return np.log2(num_classes)


print("Entropy for X:", compute_entropy(x_probs))
print("Entropy for Y:", compute_entropy(y_probs))
print("Maximal Entropy of X:", compute_max_entropy(df["x_label"]))
print("Maximal Entropy of Y:", compute_max_entropy(df["y_label"]))


# =========================================================
# 2.1.e
# Load the multi-class entropy dataset and plot X and Y
# =========================================================

df_multi = pd.read_csv("data_entropy_multi.csv")

# str.strip() removes leading/trailing spaces from column names if they exist
df_multi.columns = df_multi.columns.str.strip()

print(df_multi.head())
print(df_multi.columns.tolist())

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for label in sorted(df_multi[x_label_col].unique()):
    subset = df_multi[df_multi[x_label_col] == label]
    axes[0].scatter(subset[x_cols[0]], subset[x_cols[1]], label=label, s=20)

axes[0].set_title("Random Variable X (Multi-class)")
axes[0].set_xlabel("x_1")
axes[0].set_ylabel("x_2")
axes[0].legend()

for label in sorted(df_multi[y_label_col].unique()):
    subset = df_multi[df_multi[y_label_col] == label]
    axes[1].scatter(subset[y_cols[0]], subset[y_cols[1]], label=label, s=20)

axes[1].set_title("Random Variable Y (Multi-class)")
axes[1].set_xlabel("y_1")
axes[1].set_ylabel("y_2")
axes[1].legend()

plt.tight_layout()
plt.show()


# =========================================================
# 2.1.f
# Compute probabilities, entropy, and maximal entropy for multi-class dataset
# =========================================================

x_probs_multi = compute_probabilities(df_multi["x_label"])
y_probs_multi = compute_probabilities(df_multi["y_label"])

print("Probabilities for X:", x_probs_multi)
print("Probabilities for Y:", y_probs_multi)

print("Entropy for X:", compute_entropy(x_probs_multi))
print("Entropy for Y:", compute_entropy(y_probs_multi))
print("Maximal Entropy of X:", compute_max_entropy(df_multi["x_label"]))
print("Maximal Entropy of Y:", compute_max_entropy(df_multi["y_label"]))


# =========================================================
# 2.1.g
# Interpretation:
# More labels increase the maximal possible entropy because:
# H_max = log2(n), where n is the number of labels.
# However, the actual entropy also depends on how balanced the label
# distribution is. If labels are not equally distributed, the entropy
# will be lower than the maximal entropy.
# =========================================================


# =========================================================
# 2.2.a
# Implement Mahalanobis distance
# =========================================================

def mahalanobis_distance(mu, C_x, x):
    """
    Computes the Mahalanobis distance between a point x and a distribution
    represented by mean vector mu and covariance matrix C_x.

    Formula:
        d = sqrt( (mu - x)^T * C_x^-1 * (mu - x) )

    Input:
        mu  -> mean vector
        C_x -> covariance matrix
        x   -> point

    Output:
        distance -> scalar value
    """
    # np.array(...) converts input into a NumPy array if it is a list
    mu = np.array(mu)
    x = np.array(x)
    C_x = np.array(C_x)

    # Difference vector
    diff = mu - x

    # np.linalg.inv(C_x) computes the inverse of the covariance matrix
    C_x_inv = np.linalg.inv(C_x)

    # diff.T means transpose of the vector
    # @ means matrix multiplication
    # np.sqrt(...) computes the square root
    distance = np.sqrt(diff.T @ C_x_inv @ diff)

    return distance


# =========================================================
# 2.2.b
# Load data_mdistance.csv and compute mean vector and covariance matrix
# =========================================================

df_mdistance = pd.read_csv("data_mdistance.csv")

# Select the two feature columns and convert them to a NumPy array
# to_numpy() converts the selected DataFrame columns into a NumPy matrix
X = df_mdistance[["x_1", "x_2"]].to_numpy()

# np.mean(X, axis=0) computes the mean column-wise
# axis=0 means compute mean for each column separately
mu_hat = np.mean(X, axis=0)

# np.cov(X, rowvar=False) computes the covariance matrix
# rowvar=False means columns are variables, rows are observations
C_x_hat = np.cov(X, rowvar=False)

print("mu_hat =", mu_hat)
print("C_x_hat =\n", C_x_hat)


# =========================================================
# 2.2.c
# Compute Pearson correlation coefficient
# =========================================================

# np.corrcoef(X, rowvar=False) returns the correlation matrix
correlation_matrix = np.corrcoef(X, rowvar=False)

# [0, 1] extracts the correlation between x_1 and x_2
pearson_corr = correlation_matrix[0, 1]

print("Pearson Correlation Coefficient:", pearson_corr)


# =========================================================
# 2.2.d
# Load additional points and compute Mahalanobis and Euclidean distances
# =========================================================

df_points = pd.read_csv("data_mdistance_points.csv")
df_points.columns = df_points.columns.str.strip()

print(df_points.head())
print(df_points.columns.tolist())

# According to your dataset, the point columns are px and py
X_points = df_points[["px", "py"]].to_numpy()


def euclidean_distance(mu, x):
    """
    Computes Euclidean distance between mu and x.

    Formula:
        sqrt( sum((mu - x)^2) )

    np.linalg.norm(...) computes the vector norm,
    which for this case is the Euclidean distance.
    """
    return np.linalg.norm(mu - x)


for i, x_p in enumerate(X_points):
    d_mahal = mahalanobis_distance(mu_hat, C_x_hat, x_p)
    d_euclid = euclidean_distance(mu_hat, x_p)

    print(f"Point {i}:")
    print("  Mahalanobis distance:", d_mahal)
    print("  Euclidean distance:", d_euclid)
    print()


# =========================================================
# 2.2.e
# Plot dataset, mean vector, query points, and covariance ellipses
# =========================================================

def plot_covariance_ellipse(mu, C, ax, n_std=1.0, **kwargs):
    """
    Draws a covariance ellipse from mean vector mu and covariance matrix C.

    Steps:
    1) Compute eigenvalues and eigenvectors of covariance matrix
    2) Use them to determine ellipse orientation and size
    3) Draw ellipse on the provided axis
    """

    # np.linalg.eigh(C) computes eigenvalues and eigenvectors of a symmetric matrix
    # Covariance matrices are symmetric, so eigh is appropriate here
    eigenvalues, eigenvectors = np.linalg.eigh(C)

    # argsort() returns indices that would sort the array
    # [::-1] reverses the order so we get descending order
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # arctan2(y, x) computes the rotation angle of the main eigenvector
    # np.degrees(...) converts radians to degrees
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

    # The ellipse axes are based on sqrt(eigenvalues)
    width = 2 * n_std * np.sqrt(eigenvalues[0])
    height = 2 * n_std * np.sqrt(eigenvalues[1])

    ellipse = Ellipse(
        xy=mu,
        width=width,
        height=height,
        angle=angle,
        fill=False,
        **kwargs
    )

    # add_patch(...) adds the ellipse shape to the plot
    ax.add_patch(ellipse)


plt.figure(figsize=(8, 6))

# Plot original dataset
plt.scatter(X[:, 0], X[:, 1], label="Data points", s=20)

# Plot extra query points
plt.scatter(X_points[:, 0], X_points[:, 1], label="Query points", marker="x", s=80)

# Plot mean vector
plt.scatter(mu_hat[0], mu_hat[1], label="Mean vector", marker="o", s=120)

# Plot covariance ellipses
plot_covariance_ellipse(mu_hat, C_x_hat, plt.gca(), n_std=1, edgecolor="red", linewidth=2)
plot_covariance_ellipse(mu_hat, C_x_hat, plt.gca(), n_std=2, edgecolor="green", linewidth=2)

plt.xlabel("x_1")
plt.ylabel("x_2")
plt.title("Mahalanobis Distance Visualization")
plt.legend()

# axis("equal") makes x and y scales equal so distances/shapes look correct
plt.axis("equal")
plt.show()


# =========================================================
# 2.2.f
# Interpretation:
# Mahalanobis distance is better than Euclidean distance because it takes
# into account the covariance structure of the data, including variance
# and correlation between features. Euclidean distance treats all directions
# equally and ignores the data distribution.
# =========================================================


# =========================================================
# 2.3.b
# Compute KL divergence for the two discrete distributions
# =========================================================

# np.array(...) creates NumPy arrays from Python lists
p = np.array([0.05, 0.20, 0.40, 0.20, 0.15])
q = np.array([0.10, 0.60, 0.05, 0.15, 0.10])


def kl_divergence(p, q):
    """
    Computes Kullback-Leibler divergence:

        D_KL(P || Q) = sum( p(x) * log( p(x) / q(x) ) )

    Input:
        p -> first probability distribution
        q -> second probability distribution

    Output:
        KL divergence value
    """
    result = 0

    # len(p) gives the number of elements in the distribution
    for i in range(len(p)):
        # np.log(...) computes the natural logarithm
        result += p[i] * np.log(p[i] / q[i])

    return result


d_pq = kl_divergence(p, q)
d_qp = kl_divergence(q, p)

print("KL(P || Q):", d_pq)
print("KL(Q || P):", d_qp)


