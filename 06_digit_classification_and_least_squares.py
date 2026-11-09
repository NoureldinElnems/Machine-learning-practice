# %%
# Noureldin Elnems / noe2466@thi.de / Group 07

# 6.1.a Load digits dataset and split into train / val / test
# ============================================================

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


digits = load_digits()
X = digits.data

# y contains the correct labels.
y = digits.target

print("Original data shape:", X.shape)
print("Original labels shape:", y.shape)


# Data splitting, The temporary data will later be split into validation and test sets.

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,      # 30% = validation + test
    random_state=42,     # makes the random split reproducible
    shuffle=True         # randomizes the data before splitting
)


# Second split:
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=2/3,       # 2/3 of temp = 20% of original data
    random_state=42,
    shuffle=True
)


# Print the shapes to verify that the split is correct
print("\nAfter splitting:")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

total_samples = len(X)

# Optional: Print percentages to confirm the split ratios
'''
print("\nSplit percentages:")
print("Train:", round(len(X_train) / total_samples * 100, 2), "%")
print("Validation:", round(len(X_val) / total_samples * 100, 2), "%")
print("Test:", round(len(X_test) / total_samples * 100, 2), "%")
'''

# 6.1.b Define evaluation function
# ============================================================

def evaluate(y_pred, y_gt):

    # Convert both inputs to NumPy arrays.
    y_pred = np.array(y_pred)
    y_gt = np.array(y_gt)

    # Compare predictions with ground truth.
    correct_predictions = (y_pred == y_gt)
    accuracy = np.sum(correct_predictions) / len(y_gt)

    return accuracy

# Test the evaluate function with a simple example
'''
y_pred_test = np.array([1, 2, 3, 4])
y_gt_test = np.array([1, 2, 8, 4])

acc = evaluate(y_pred_test, y_gt_test)

print("Accuracy:", acc)
print("Accuracy percentage:", acc * 100, "%")
'''

# 6.1.c Implement PCA and reduce data from 64D to 4D
# ============================================================

def fit_pca(X_train, n_components):

    # Compute the mean vector of the training data
    mean_vector = np.mean(X_train, axis=0)

    # Center the training data, PCA should be applied to centered data
    X_train_centered = X_train - mean_vector

    # Compute the covariance matrix / rowvar=False means:columns are features, rows are samples.
    covariance_matrix = np.cov(X_train_centered, rowvar=False)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Sort eigenvectors by eigenvalues in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Select the first n_components eigenvectors
    principal_components = eigenvectors[:, :n_components]

    return mean_vector, principal_components


def transform_pca(X, mean_vector, principal_components):
 
    X_centered = X - mean_vector
    X_reduced = X_centered @ principal_components
    return X_reduced

# Fit PCA on the training data only
mean_4d, components_4d = fit_pca(X_train, n_components=4)

# Apply the same PCA transformation to all datasets
X_train_pca4 = transform_pca(X_train, mean_4d, components_4d)
X_val_pca4 = transform_pca(X_val, mean_4d, components_4d)
X_test_pca4 = transform_pca(X_test, mean_4d, components_4d)

print("Original X_train shape:", X_train.shape)
print("Reduced X_train shape:", X_train_pca4.shape)

print("Original X_val shape:", X_val.shape)
print("Reduced X_val shape:", X_val_pca4.shape)

print("Original X_test shape:", X_test.shape)
print("Reduced X_test shape:", X_test_pca4.shape)


# 6.1.d Implement k-NN classifier and choose the best k
# ============================================================

from collections import Counter
import pandas as pd

def predict_knn(X_train, y_train, X_query, k):

    # List to store predicted labels
    y_pred = []

    # Loop over every query sample
    for query_sample in X_query:

        # Compute distances from the query sample to all training samples
        distances = np.linalg.norm(X_train - query_sample, axis=1)

        # Step 2: Get indices of the k smallest distances
        nearest_indices = np.argsort(distances)[:k]

        nearest_labels = y_train[nearest_indices]

        # The most common label among the nearest neighbors becomes the prediction
        most_common_label = Counter(nearest_labels).most_common(1)[0][0]

        # Save prediction
        y_pred.append(most_common_label)

    return np.array(y_pred)



# Hyperparameter values required in the exercise
k_values = [1, 4, 8, 16]

# List to store results for the final table
validation_results = []

# Test each k value using the validation dataset
for k in k_values:
    y_val_pred = predict_knn(
        X_train_pca4,
        y_train,
        X_val_pca4,
        k
    )

    val_accuracy = evaluate(y_val_pred, y_val)

    validation_results.append({
        "k": k,
        "Validation Accuracy": val_accuracy,
        "Validation Accuracy (%)": val_accuracy * 100
    })

# Convert results into a table
validation_table = pd.DataFrame(validation_results)
print(validation_table)


# 6.1.e Evaluate the k-NN classifier on the test dataset
# ============================================================

# From 6.1.d, the highest validation accuracy was achieved at k = 8

best_k = 8

# Predict the labels of the PCA-reduced test dataset
y_test_pred = predict_knn(
    X_train_pca4,   
    y_train,        
    X_test_pca4,    
    best_k          
)

test_accuracy = evaluate(y_test_pred, y_test)


print("Best k selected from validation:", best_k)
print("Test accuracy (%):", test_accuracy * 100)


# 6.1.f Reduce test data to 2D with PCA and visualize results
# ============================================================

import matplotlib.pyplot as plt

# Fit PCA with 2 components on the training data only
mean_2d, components_2d = fit_pca(X_train, n_components=2)
X_test_pca2 = transform_pca(X_test, mean_2d, components_2d)

print("Original X_test shape:", X_test.shape)
print("Reduced X_test shape:", X_test_pca2.shape)

# Create one figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1: Test data colored by predicted labels
scatter_pred = axes[0].scatter(
    X_test_pca2[:, 0],     # first PCA component
    X_test_pca2[:, 1],     # second PCA component
    c=y_test_pred,         # colors are based on predicted labels
    cmap="tab10",
    s=35
)

axes[0].set_title("Test Dataset - Predicted Labels")
axes[0].set_xlabel("Principal Component 1")
axes[0].set_ylabel("Principal Component 2")
axes[0].grid(True)


# Subplot 2: Test data colored by ground truth labels
scatter_gt = axes[1].scatter(
    X_test_pca2[:, 0],     # first PCA component
    X_test_pca2[:, 1],     # second PCA component
    c=y_test,              # colors are based on true labels
    cmap="tab10",
    s=35
)

axes[1].set_title("Test Dataset - Ground Truth Labels")
axes[1].set_xlabel("Principal Component 1")
axes[1].set_ylabel("Principal Component 2")
axes[1].grid(True)


# Add colorbars to show digit labels from 0 to 9
cbar_pred = fig.colorbar(scatter_pred, ax=axes[0])
cbar_pred.set_label("Predicted digit label")

cbar_gt = fig.colorbar(scatter_gt, ax=axes[1])
cbar_gt.set_label("Ground truth digit label")


# Improve spacing and show plot
plt.tight_layout()
plt.show()


