# %%
#author: Noureldin Elnems  , noe2466@thi.de , Group: 7 
#        Md Rasel , mdr3149@thi.de , Group: 7

# EXERCISE 1: Decision Tree Classification
# ============================================================================

# Imports needed for Question 1
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.inspection import DecisionBoundaryDisplay


# 8.1.a Load the dataset data_exercise_1.pkl
# ============================================================================

with open("data_exercise_1.pkl", "rb") as file:
    data = pickle.load(file)


# We assume that:
# - the first two columns are the input features
# - the third column is the class label

X = data.iloc[:, :2].values
y = data.iloc[:, 2].values

# 8.1.b Split the dataset into train and test subsets
# ============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.5,
    random_state=42,
    stratify=y
)

# 8.1.c Implement and train a Decision Tree using sklearn
# ============================================================================

# Create a Decision Tree classifier.
dt = DecisionTreeClassifier(random_state=42)

# Train the decision tree on the training data.
dt.fit(X_train, y_train)

# 8.1.d Predict the test labels and evaluate using F-Score
# ============================================================================

# Predict the labels of the test dataset.
y_pred = dt.predict(X_test)

# Calculate the F-Score.
fscore = f1_score(y_test, y_pred, average="weighted")

print(f"F-Score: {fscore:.4f}")

# 8.1.e Plot train/test data and the learned decision boundary
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left subplot: train dataset
# ---------------------------------------------------------------------------
DecisionBoundaryDisplay.from_estimator(
    dt,
    X,                         # use full X so both plots have the same boundary range
    response_method="predict",
    alpha=0.3,
    ax=ax1,
    cmap="coolwarm"
)

ax1.scatter(
    X_train[:, 0],
    X_train[:, 1],
    c=y_train,
    cmap="coolwarm",
    edgecolors="black",
    s=40
)

ax1.set_title("Train Dataset - Ground Truth Labels")
ax1.set_xlabel("x1")
ax1.set_ylabel("x2")

# Right subplot: test dataset
# ----------------------------------------------------------------------------
DecisionBoundaryDisplay.from_estimator(
    dt,
    X,                         # same full X range for visual consistency
    response_method="predict",
    alpha=0.3,
    ax=ax2,
    cmap="coolwarm"
)

ax2.scatter(
    X_test[:, 0],
    X_test[:, 1],
    c=y_test,
    cmap="coolwarm",
    edgecolors="black",
    s=40
)

ax2.set_title("Test Dataset - Ground Truth Labels")
ax2.set_xlabel("x1")
ax2.set_ylabel("x2")
plt.tight_layout()
plt.show()

# %%
# EXERCISE 2: Ensemble Method - Random Forest Algorithm
# ============================================================================

# Imports needed for Question 2
# ----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss
from sklearn.utils import resample

# 8.2.a Load the dataset data_exercise_2.pkl
# ============================================================================
with open("data_exercise_2.pkl", "rb") as file:
    data2 = pickle.load(file)

X2 = data2.iloc[:, :2].values
y2 = data2.iloc[:, 2].values

# Get all class labels in the dataset.
classes = np.unique(y2)
num_classes = len(classes)

# 8.2.b Generate and train N = 30 different decision trees
# ============================================================================

N = 30
total_samples = len(X2)

samples_per_tree = max(1, int(total_samples * 0.01))

trees = []

for i in range(N):

    X_boot, y_boot = resample(
        X2,
        y2,
        replace=True,
        n_samples=samples_per_tree,
        random_state=i
    )

    tree = DecisionTreeClassifier(random_state=i)
    tree.fit(X_boot, y_boot)
    trees.append(tree)

# 8.2.c Split the dataset into train and test subsets for evaluation
# ============================================================================
X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X2,
    y2,
    test_size=0.5,
    random_state=42,
    stratify=y2
)

# 8.2.d Predict using all trees and apply majority voting
# ============================================================================

tree_predictions = []

for tree in trees:
    prediction = tree.predict(X_test2)
    tree_predictions.append(prediction)

tree_predictions = np.array(tree_predictions)

# Majority voting
# ----------------------------------------------------------------------------

y_pred_ensemble = []
ensemble_probabilities = []

for sample_index in range(tree_predictions.shape[1]):

    # Predictions of all trees for the current test sample.
    votes = tree_predictions[:, sample_index]

    # Count votes for each class.
    class_vote_counts = []

    for class_label in classes:
        vote_count = np.sum(votes == class_label)
        class_vote_counts.append(vote_count)

    class_vote_counts = np.array(class_vote_counts)

    class_probabilities = class_vote_counts / N

    # The final predicted class is the class with the highest probability.
    predicted_class = classes[np.argmax(class_probabilities)]

    y_pred_ensemble.append(predicted_class)
    ensemble_probabilities.append(class_probabilities)

y_pred_ensemble = np.array(y_pred_ensemble)
ensemble_probabilities = np.array(ensemble_probabilities)

# 8.2.e Evaluate prediction performance using Cross-Entropy loss
# ============================================================================

cross_entropy_loss = log_loss(
    y_test2,
    ensemble_probabilities,
    labels=classes
)

print(f"Cross-Entropy Loss: {cross_entropy_loss:.4f}")

# 8.2.f Plot train dataset and test predicted label probability
# ============================================================================

if num_classes == 2:
    test_probability_to_plot = ensemble_probabilities[:, 1]
    probability_label = f"Predicted probability of class {classes[1]}"
else:
    # For multi-class data, plot confidence of the predicted class.
    test_probability_to_plot = np.max(ensemble_probabilities, axis=1)
    probability_label = "Predicted class probability"

# Create two subplots.

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))


# Left subplot: train dataset with ground truth labels
# ----------------------------------------------------------------------------

scatter1 = ax1.scatter(
    X_train2[:, 0],
    X_train2[:, 1],
    c=y_train2,
    cmap="coolwarm",
    edgecolors="black",
    s=40
)

ax1.set_title("Train Dataset - Ground Truth Labels")
ax1.set_xlabel("x1")
ax1.set_ylabel("x2")


# Right subplot: test dataset with predicted label probability
# ----------------------------------------------------------------------------

scatter2 = ax2.scatter(
    X_test2[:, 0],
    X_test2[:, 1],
    c=test_probability_to_plot,
    cmap="coolwarm",
    edgecolors="black",
    s=40,
    vmin=0,
    vmax=1
)

ax2.set_title("Test Dataset - Ensemble Predicted Probability")
ax2.set_xlabel("x1")
ax2.set_ylabel("x2")


# Add colorbar to explain the probability colors.

colorbar = plt.colorbar(scatter2, ax=ax2)
colorbar.set_label(probability_label)


# Final title and layout.

fig.suptitle("Ensemble Method Prediction", fontsize=14)

plt.tight_layout()
plt.show()

# %%
# EXERCISE 3: Gini Impurity
# ============================================================================

# Imports needed for Question 3
import numpy as np
import matplotlib.pyplot as plt
import pickle

# 8.3.a Load the dataset gini_imp.pkl
# ============================================================================
with open("gini_imp.pkl", "rb") as file:
    gini_data = pickle.load(file)

if hasattr(gini_data, "iloc"):

    # If it is a pandas DataFrame:
    X_gini = gini_data.iloc[:, :2].values
    y_gini = gini_data.iloc[:, 2].values

else:

    # If it is already a NumPy array:
    gini_data = np.array(gini_data)
    X_gini = gini_data[:, :2]
    y_gini = gini_data[:, 2]

# 8.3.b Write a function called gini_impurity()
# ============================================================================

def gini_impurity(y):
    y = np.array(y)
    if len(y) == 0:
        return 0
    unique_classes, class_counts = np.unique(y, return_counts=True)
    class_probabilities = class_counts / len(y)
    impurity = 1 - np.sum(class_probabilities ** 2)

    return impurity

# 8.3.c Write a function called split_data()
# ============================================================================

def split_data(X, y, xdiv):

    # Create mask for set 1.
    set1_mask = X[:, 0] <= xdiv

    # Create mask for set 2.
    set2_mask = X[:, 0] > xdiv

    # Apply the masks.
    X_set1 = X[set1_mask]
    y_set1 = y[set1_mask]

    X_set2 = X[set2_mask]
    y_set2 = y[set2_mask]

    return X_set1, y_set1, X_set2, y_set2

# 8.3.d Write a function called split_gini_impurity()
# ============================================================================

def split_gini_impurity(X, y, xdiv):

    # Split the data.
    X_set1, y_set1, X_set2, y_set2 = split_data(X, y, xdiv)

    # Total number of samples.
    total_size = len(y)

    # Sizes of the two subsets.
    size_set1 = len(y_set1)
    size_set2 = len(y_set2)

    # Gini Impurity of each subset.
    gini_set1 = gini_impurity(y_set1)
    gini_set2 = gini_impurity(y_set2)

    # Weighted Gini Impurity.
    total_gini = (size_set1 / total_size) * gini_set1 + \
                 (size_set2 / total_size) * gini_set2

    return total_gini

# 8.3.e Compute Gini Impurity for given split positions and plot results
# ============================================================================

# Split positions given in the exercise
xdiv_values = [-5, -2, 0, 1, 10]

# Store computed Gini values here
gini_values = []

# Compute Gini Impurity for each split position.
for xdiv in xdiv_values:

    impurity_value = split_gini_impurity(X_gini, y_gini, xdiv)
    gini_values.append(impurity_value)

    print(f"xdiv = {xdiv:>3} --> Gini Impurity = {impurity_value:.4f}")


# Plotting

fig, axes = plt.subplots(1, 5, figsize=(20, 4))

fig.suptitle("Gini Impurities", fontsize=14)

for ax, xdiv, impurity_value in zip(axes, xdiv_values, gini_values):

    # Plot the original data.
    # Color represents the true class label.
    ax.scatter(
        X_gini[:, 0],
        X_gini[:, 1],
        c=y_gini,
        cmap="viridis",
        edgecolors="black",
        s=40
    )

    # Draw the split position as a red dashed vertical line.

    ax.axvline(
        x=xdiv,
        color="red",
        linestyle="--",
        linewidth=2
    )

    # Add the computed Gini Impurity to the title.

    ax.set_title(f"Impurity = {impurity_value:.3f}")

    # Axis labels.

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")


plt.tight_layout()
plt.show()


