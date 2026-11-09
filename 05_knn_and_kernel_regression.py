# %%
# Noureldin Elnems / noe2466@thi.de / Group 07

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 5.1.a k-Nearest Neighbors Classifier

class KNeighborsClassifier:

    def __init__(self, n_neighbors=5, distance_metric=None):
        self.n_neighbors = n_neighbors
        self.distance_metric = distance_metric

    def fit(self, X_train, y_train):

        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
 
        predictions = []

        # Loop over every test point
        for xi in X_test:

            distances = self.distance_metric(xi, self.X_train)  
            nearest_indices = np.argsort(distances)[:self.n_neighbors]

            nearest_labels = self.y_train[nearest_indices]

            labels, counts = np.unique(nearest_labels, return_counts=True)

            predicted_label = labels[np.argmax(counts)]

            predictions.append(predicted_label)

        return np.array(predictions)





# 5.1.b Euclidean distance function

def euclidean_distance(xi, X):

    distances = np.sqrt(np.sum((X - xi) ** 2, axis=1))

    return distances




# 5.1.c Evaluation function

def evaluate(y_pred, y_gt):

    correct_predictions = np.sum(y_pred == y_gt)
    accuracy = correct_predictions / len(y_gt)

    return accuracy




# 5.1.d Load the datasets

# The files are in the same folder as the notebook/script, Each file contains train and test data.
data_blobs = pd.read_pickle("data_blobs.pkl")
data_moons = pd.read_pickle("data_moons.pkl")
data_full = pd.read_pickle("data_full.pkl")

# Print the content structure to understand the keys/columns
print("Blobs data:")
print(data_blobs)

print("\nMoons data:")
print(data_moons)

print("\nFull data:")
print(data_full)





# 5.1.e Train k-NN classifiers with k = 1, 15, 30

def split_dataset(data):


    train_data = data.dropna(subset=["x1_train", "x2_train", "y_train"])

    X_train = train_data[["x1_train", "x2_train"]].to_numpy()
    y_train = train_data["y_train"].to_numpy()


    test_data = data.dropna(subset=["x1_test", "x2_test", "y_test"])

    X_test = test_data[["x1_test", "x2_test"]].to_numpy()
    y_test = test_data["y_test"].to_numpy()

    return X_train, y_train, X_test, y_test


# Store datasets for looping
datasets = {
    "Blobs": data_blobs,
    "Moons": data_moons,
    "Full": data_full
}

k_values = [1, 15, 30]

# Store results for point (f)
results = {}

for dataset_name, data in datasets.items():

    X_train, y_train, X_test, y_test = split_dataset(data)

    results[dataset_name] = {}

    print(f"\nResults for {dataset_name} dataset:")

    for k in k_values:

        # Create model
        knn = KNeighborsClassifier(
            n_neighbors=k,
            distance_metric=euclidean_distance
        )

        knn.fit(X_train, y_train)

        y_pred = knn.predict(X_test)

        accuracy = evaluate(y_pred, y_test)

        # Save everything for later plotting
        results[dataset_name][k] = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": y_pred,
            "accuracy": accuracy
        }

        print(f"k = {k}: accuracy = {accuracy:.4f}")





# 5.1.f Plot ground truth and predictions for k = 15

k_plot = 15

fig, axes = plt.subplots(2, 3, figsize=(12, 7))

fig.suptitle("k-NN Performance on Different Datasets (k = 15)", fontsize=14)

for col, dataset_name in enumerate(datasets.keys()):

    # Get saved data from point (e)
    X_train = results[dataset_name][k_plot]["X_train"]
    y_train = results[dataset_name][k_plot]["y_train"]
    X_test = results[dataset_name][k_plot]["X_test"]
    y_test = results[dataset_name][k_plot]["y_test"]
    y_pred = results[dataset_name][k_plot]["y_pred"]

    # First row: ground truth

    axes[0, col].scatter(
        X_train[:, 0],
        X_train[:, 1],
        c=y_train,
        marker="x",
        label="train GT"
    )

    axes[0, col].scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=y_test,
        marker=">",
        label="test GT"
    )

    axes[0, col].set_title(f"{dataset_name} GT")
    axes[0, col].set_xlabel("x")
    axes[0, col].set_ylabel("y")
    axes[0, col].legend(fontsize=7)

    # Second row: prediction

    axes[1, col].scatter(
        X_train[:, 0],
        X_train[:, 1],
        c=y_train,
        marker="x",
        label="train GT"
    )

    axes[1, col].scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=y_pred,
        marker=">",
        label="test pred"
    )

    axes[1, col].set_title(f"{dataset_name} pred")
    axes[1, col].set_xlabel("x")
    axes[1, col].set_ylabel("y")
    axes[1, col].legend(fontsize=7)

plt.tight_layout()
plt.show()


