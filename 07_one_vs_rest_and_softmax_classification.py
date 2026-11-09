# %%
# Noureldin Elnems / noe2466@thi.de / Group 07

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron

# 7.1.a Load the digits dataset and generate train/test subsets
# ============================================================

# Load the handwritten digits dataset
digits = load_digits()
X = digits.data
y = digits.target

# Split the dataset into: 70% training data & 30% test data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# Print the original data shapes
print("Original data shapes:")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# 7.1.b Implement PCA algorithm
# ============================================================

def fit_pca(X_train, n_components):
    
    # Step 1: Compute the mean vector
    mean_vector = np.mean(X_train, axis=0)

    # Step 2: Center the training data
    X_train_centered = X_train - mean_vector

    # Step 3: Compute the covariance matrix
    covariance_matrix = np.cov(X_train_centered, rowvar=False)

    # Step 4: Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Step 5: Sort eigenvectors by eigenvalues in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]

    # Reorder eigenvalues and eigenvectors
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Step 6: Select the first n_components eigenvectors
    principal_components = eigenvectors[:, :n_components]

    return mean_vector, principal_components


def transform_pca(X, mean_vector, principal_components):

    # Center the data using the training mean
    X_centered = X - mean_vector

    # Project the data onto the selected principal components
    X_reduced = X_centered @ principal_components

    return X_reduced


# Apply PCA to reduce dimensionality from R^64 to R^4
# ------------------------------------------------------------

n_components_4d = 4

# Fit PCA only on the training data
mean_vector_4d, principal_components_4d = fit_pca(X_train, n_components_4d)

# Apply the same PCA transformation to training and test data
X_train_pca_4d = transform_pca(X_train, mean_vector_4d, principal_components_4d)
X_test_pca_4d = transform_pca(X_test, mean_vector_4d, principal_components_4d)

# Print PCA-reduced shapes
print("\nAfter PCA to 4 dimensions:")
print("X_train_pca_4d shape:", X_train_pca_4d.shape)
print("X_test_pca_4d shape:", X_test_pca_4d.shape)


# 7.1.c Implement One-vs-Rest multiclass classifier
# ============================================================

class OneVsRest:

    def __init__(self, K):
       
        # Store the number of classes
        self.K = K
        self.classifiers = []

        # Create K linear binary classifiers
        for _ in range(K):
            classifier = Perceptron(
                max_iter=1000,
                tol=1e-3,
                random_state=42
            )
            # Add the classifier to the list
            self.classifiers.append(classifier)

    def fit(self, X, y):

        # Train one binary classifier for each class
        for class_label in range(self.K):
            y_binary = np.where(y == class_label, 1, -1)

            # Train the binary classifier for the current class
            self.classifiers[class_label].fit(X, y_binary)

    def predict(self, X):

        # This list will store the scores of all classifiers
        all_scores = []

        # Get a score from each binary classifier
        for classifier in self.classifiers:
            scores = classifier.decision_function(X)
            all_scores.append(scores)

        # Convert list to NumPy array and transpose to shape (num_samples, K)
        all_scores = np.array(all_scores).T
        y_pred = np.argmax(all_scores, axis=1)

        return y_pred


# 7.1.d Train One-vs-Rest classifier with K = 10
# ============================================================

K = 10
ovr_classifier = OneVsRest(K)

# Train the classifier using the 4D PCA training data
ovr_classifier.fit(X_train_pca_4d, y_train)

y_pred_4d = ovr_classifier.predict(X_test_pca_4d)

print("\nFirst 20 predicted labels using 4D PCA:")
print(y_pred_4d[:20])
print("\nFirst 20 true labels:")
print(y_test[:20])


# 7.1.e Evaluate the classifier
# ============================================================

def evaluate(y_pred, y_gt):
    correct_predictions = y_pred == y_gt
    accuracy = np.mean(correct_predictions)
    return accuracy


# Evaluate the 4D PCA classifier
accuracy_4d = evaluate(y_pred_4d, y_test)
print("\nAccuracy percentage using 4D PCA:")
print(accuracy_4d * 100, "%")


# 7.1.f Apply PCA to reduce data to 2 dimensions and plot results
# ============================================================

# Number of dimensions required for visualization
n_components_2d = 2

# Fit PCA again on the original training data, but now keep only 2 components
mean_vector_2d, principal_components_2d = fit_pca(X_train, n_components_2d)

# Transform training and test data to 2D
X_train_pca_2d = transform_pca(X_train, mean_vector_2d, principal_components_2d)
X_test_pca_2d = transform_pca(X_test, mean_vector_2d, principal_components_2d)

# Print PCA 2D shapes
print("\nAfter PCA to 2 dimensions:")
print("X_train_pca_2d shape:", X_train_pca_2d.shape)
print("X_test_pca_2d shape:", X_test_pca_2d.shape)

# Create a new One-vs-Rest classifier for the 2D data
ovr_classifier_2d = OneVsRest(K)

# Train it using the 2D PCA training data
ovr_classifier_2d.fit(X_train_pca_2d, y_train)

# Predict labels for the 2D PCA test data
y_pred_2d = ovr_classifier_2d.predict(X_test_pca_2d)

# Evaluate the 2D PCA classifier
accuracy_2d = evaluate(y_pred_2d, y_test)

print("\nAccuracy percentage using 2D PCA:")
print(accuracy_2d * 100, "%")


# Plot ground truth labels and predicted labels
# ============================================================

# Create one figure with two subplots side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 6))


# First subplot: Ground truth labels
# ------------------------------------------------------------
scatter1 = axes[0].scatter(
    X_test_pca_2d[:, 0],   # First PCA component
    X_test_pca_2d[:, 1],   # Second PCA component
    c=y_test,              # Color points by true labels
    cmap="tab10",
    s=30
)

axes[0].set_title("Ground Truth Labels")
axes[0].set_xlabel("First Principal Component")
axes[0].set_ylabel("Second Principal Component")

# Add a colorbar to show digit classes
plt.colorbar(scatter1, ax=axes[0])


# Second subplot: Predicted labels
# ------------------------------------------------------------
scatter2 = axes[1].scatter(
    X_test_pca_2d[:, 0],   # First PCA component
    X_test_pca_2d[:, 1],   # Second PCA component
    c=y_pred_2d,           # Color points by predicted labels
    cmap="tab10",
    s=30
)

axes[1].set_title("Predicted Labels - One-vs-Rest")
axes[1].set_xlabel("First Principal Component")
axes[1].set_ylabel("Second Principal Component")

# Add a colorbar to show predicted digit classes
plt.colorbar(scatter2, ax=axes[1])

# Add one main title for the whole figure
fig.suptitle("Digits Dataset Classification using PCA + One-vs-Rest", fontsize=14)

# Adjust spacing between plots
plt.tight_layout()

# Show the plot
plt.show()


