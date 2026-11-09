# %%
import sys

# Install sklearn into the SAME environment used by this notebook
!{sys.executable} -m pip install scikit-learn

# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 3.2.a Load the digits dataset and plot one example for each digit 0-9
# ============================================================


from sklearn.datasets import load_digits

# Load the built-in handwritten digits dataset
digits = load_digits()

# V: data matrix of shape (M, N)
# M = number of samples
# N = number of features per sample
V = digits.data

# labels: digit class for each sample
labels = digits.target

# images: original 8x8 image form
images = digits.images

# Create one figure with multiple axes to show one example of each digit
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
fig.suptitle("Examples for Digits 0-9")

# Flatten axes array for easier indexing
axes = axes.ravel()

for digit in range(10):
    # Find the first sample index corresponding to this digit
    idx = np.where(labels == digit)[0][0]

    # Show the image
    axes[digit].imshow(images[idx], cmap="viridis")
    axes[digit].set_title(f"Digit {digit}")
    axes[digit].axis("off")

plt.tight_layout()
plt.show()

# 3.2.b Print the number of samples M and the number of features N
# ============================================================

M, N = V.shape

print("Number of samples M =", M)
print("Number of features N =", N)


# 3.2.c(i) Compute the feature-wise mean vector and center the data
# ============================================================

# mu_hat has shape (N,)
mu_hat = np.mean(V, axis=0)

V_centered = V - mu_hat

print("\nShape of mean vector mu_hat:", mu_hat.shape)
print("Shape of centered data V_centered:", V_centered.shape)

print("First 10 centered feature means:")
print(np.mean(V_centered, axis=0)[:10])


# 3.2.c(ii) Compute the sample covariance matrix C_v
# ============================================================

C_v = (1 / M) * (V_centered.T @ V_centered)
print("\nShape of covariance matrix C_v:", C_v.shape)
print("Top-left 5x5 block of C_v:")
print(C_v[:5, :5])


# 3.2.c(iii) Perform eigendecomposition
# ============================================================
eigenvalues, U = np.linalg.eig(C_v)
print("\nShape of eigenvalues:", eigenvalues.shape)
print("Shape of eigenvector matrix U:", U.shape)
print("First 10 eigenvalues:")
print(eigenvalues[:10])


# ============================================================
# 3.2.c(iv) Explain what U denotes
# ============================================================

print("\nU is the matrix of eigenvectors of the covariance matrix C_v.")
print("Each column of U is one eigenvector.")
print("Each eigenvector represents one principal direction in the feature space.")
print("Its corresponding eigenvalue represents the amount of variance captured in that direction.")

# 3.2.c(v) Select K = 2 eigenvectors corresponding to the largest eigenvalues
# ============================================================

K = 2
sorted_indices = np.argsort(eigenvalues)[::-1]

eigenvalues_sorted = eigenvalues[sorted_indices]
U_sorted = U[:, sorted_indices]

# Keep the top K eigenvectors
U_k = U_sorted[:, :K]

print("\nTop 2 eigenvalues:")
print(eigenvalues_sorted[:K])
print("Shape of U_k:", U_k.shape)


# 3.2.c(vi) Perform dimensionality reduction
# ============================================================

V_proj = V_centered @ U_k
print("\nShape of projected data V_proj:", V_proj.shape)
print("First 5 projected samples:")
print(V_proj[:5])


# 3.2.c(vii) Plot the projected parameter space with labels
# ============================================================

plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    V_proj[:, 0],
    V_proj[:, 1],
    c=labels,
    cmap="viridis",
    s=15,
    alpha=0.7
)

plt.title("Principal Components of Digits (K=2)")
plt.xlabel("component 1")
plt.ylabel("component 2")
plt.colorbar(scatter, label="digit label")
plt.grid(True)
plt.show()

# %%
# 3.3.a Implement compute_p_j_given_m()
# ============================================================

def compute_p_j_given_m(V, m, sigma_m):

    # Number of samples

    M = V.shape[0]
    p_j_given_m = np.zeros(M, dtype=float)
  
    if sigma_m <= 0:
        raise ValueError("sigma_m must be positive.")

    # Get the reference point v_m
    v_m = V[m]

    # Step 1: Compute the unnormalized Gaussian scores
    # ------------------------------------------------------------
  
    for j in range(M):
        if j == m:
            continue

        diff = v_m - V[j]
        squared_distance = np.sum(diff ** 2)
        p_j_given_m[j] = np.exp(-squared_distance / (2 * (sigma_m ** 2)))

    # Step 2: Normalize the scores into probabilities
    # ------------------------------------------------------------
    denominator = np.sum(p_j_given_m)
    if denominator == 0:
        return p_j_given_m

    p_j_given_m = p_j_given_m / denominator

    return p_j_given_m

# 3.3.b Implement compute_perplexity()
# ============================================================

def compute_perplexity(p_j_given_m):
    p_nonzero = p_j_given_m[p_j_given_m > 0]
    entropy = -np.sum(p_nonzero * np.log2(p_nonzero))
    perplexity = 2 ** entropy

    return perplexity, entropy

# 3.3(c) Implement binary_search()
# ============================================================

def binary_search(V, m, target_perplexity, lmin=0.0, lmax=100.0, I=100, eps=0.001):
  
    left = max(lmin, 1e-12)
    right = lmax

    best_sigma = None
    best_perplexity = None
    best_p = None
    smallest_error = float("inf")

    for iteration in range(I):
        # Stop if the interval becomes sufficiently small
        if (right - left) <= eps:
            break

        # Midpoint of current interval
        sigma_m = (left + right) / 2.0

        # Compute probabilities for this sigma
        p_j_given_m = compute_p_j_given_m(V, m, sigma_m)

        # Compute resulting perplexity
        perplexity, _ = compute_perplexity(p_j_given_m)

        # Measure how close we are to the target
        error = abs(perplexity - target_perplexity)

        # Save the best result seen so far
        if error < smallest_error:
            smallest_error = error
            best_sigma = sigma_m
            best_perplexity = perplexity
            best_p = p_j_given_m

        # If the perplexity is close enough, stop early
        if error <= eps:
            break

        # Binary search update:
        # If current perplexity is too small, sigma must increase
        if perplexity < target_perplexity:
            left = sigma_m
        else:
            right = sigma_m

    return best_sigma, best_perplexity, best_p

# 3.3.d Load tsne_data.pkl and determine sigma_m for each sample
# ============================================================

data = pd.read_pickle("tsne_data.pkl")

if isinstance(data, pd.DataFrame):
    V = data.to_numpy(dtype=float)
elif isinstance(data, pd.Series):
    V = data.to_numpy(dtype=float)
else:
    V = np.array(data, dtype=float)

if V.ndim != 2:
    raise ValueError("Loaded data is not a 2D matrix.")

M, N = V.shape

print("Loaded data matrix V:")
print(V)
print("\nShape of V:", V.shape)

# ------------------------------------------------------------
# Required settings from the assignment
# ------------------------------------------------------------
target_perplexity = 10
lmin = 0.15
lmax = 50
I = 100
eps = 0.001

# ------------------------------------------------------------
# Find sigma_m for each sample v_m
# ------------------------------------------------------------
sigmas = []
perplexities = []
probability_vectors = []

for m in range(M):
    sigma_m, perp_m, p_m = binary_search(
        V=V,
        m=m,
        target_perplexity=target_perplexity,
        lmin=lmin,
        lmax=lmax,
        I=I,
        eps=eps
    )

    sigmas.append(sigma_m)
    perplexities.append(perp_m)
    probability_vectors.append(p_m)

# Convert results to NumPy arrays for easier printing and analysis
sigmas = np.array(sigmas, dtype=float)
perplexities = np.array(perplexities, dtype=float)
probability_vectors = np.array(probability_vectors, dtype=float)

# ------------------------------------------------------------
# Print the final results
# ------------------------------------------------------------
print("\nOptimized sigma_m for each sample:")
for m in range(M):
    print(f"Sample {m}: sigma_m = {sigmas[m]:.6f}, perplexity = {perplexities[m]:.6f}")

print("\nAll sigma values:")
print(sigmas)

print("\nProbability matrix shape (M x M):")
print(probability_vectors.shape)

print("\nConditional probability matrix P where row m contains p_{j|m}:")
print(probability_vectors)


# %%



