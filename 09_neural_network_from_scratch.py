# %%
# author: Noureldin Elnems, noe2466@thi.de, Group: 7
#         Md Rasel, mdr3149@thi.de, Group: 7

import pickle
import numpy as np
from sklearn.model_selection import train_test_split


# 9.1.a Load the dataset
# ============================================================

with open("data_exercise_mlp.pkl", "rb") as file:
    data = pickle.load(file)

# The pickle file is a pandas DataFrame.
# The first columns are the input features, and the last column is the output label.
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values.reshape(-1, 1)

print("X shape:", X.shape)
print("y shape:", y.shape)


# 9.1.b Split the dataset into training and testing subsets 80% training data, 20% testing data
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    train_size=0.8,
    random_state=42,
    shuffle=True
)


# 9.1.d Binary Cross-Entropy Loss Function
# ============================================================
# Written before part (c) because the function may be needed later when training the neural network.

def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    loss = -np.mean(
        y_true * np.log(y_pred)
        + (1 - y_true) * np.log(1 - y_pred)
    )

    return loss


# 9.1.c Implement the NeuralNetwork class
# ============================================================

class NeuralNetwork:
    def __init__(self, feat_dim, hidden_dim, out_dim, lr):
        self.lr = lr

        self.W1 = np.random.uniform(-1, 1, (feat_dim, hidden_dim))
        self.b1 = np.zeros((1, hidden_dim))

        self.W2 = np.random.uniform(-1, 1, (hidden_dim, out_dim))
        self.b2 = np.zeros((1, out_dim))

    def forward(self, X):
        self.X = X

        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = 1 / (1 + np.exp(-self.z1))

        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.y = 1 / (1 + np.exp(-self.z2))

        return self.y

    def backward(self, y_true):
        m = y_true.shape[0]

        dz2 = self.y - y_true
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        dz1 = np.dot(dz2, self.W2.T) * self.a1 * (1 - self.a1)
        dW1 = np.dot(self.X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        gradients = {
            "dW1": dW1,
            "db1": db1,
            "dW2": dW2,
            "db2": db2
        }

        return gradients

    def update_weights(self, gradients):
        self.W1 -= self.lr * gradients["dW1"]
        self.b1 -= self.lr * gradients["db1"]

        self.W2 -= self.lr * gradients["dW2"]
        self.b2 -= self.lr * gradients["db2"]


# 9.1.e Instantiate the NeuralNetwork class Learning rate = 0.001
# ============================================================

feat_dim = X_train.shape[1]
hidden_dim = 3
out_dim = 1
lr = 0.001

model = NeuralNetwork(feat_dim, hidden_dim, out_dim, lr)


# %%
# 9.1.f Train for E = 100 epochs and plot the loss curve
# ============================================================

import matplotlib.pyplot as plt

epochs = 100
loss_curve = []

for epoch in range(epochs):
    epoch_loss = 0

    for i in range(len(X_train)):
        x_i = X_train[i:i+1]
        y_i = y_train[i:i+1]

        y_pred = model.forward(x_i)
        loss = binary_cross_entropy(y_i, y_pred)
        epoch_loss += loss

        gradients = model.backward(y_i)
        model.update_weights(gradients)

    loss_curve.append(epoch_loss / len(X_train))

plt.figure(figsize=(7, 4))
plt.plot(loss_curve, color="steelblue")
plt.xlabel("Epoch")
plt.ylabel("BCE Loss")
plt.title("Training Loss Curve")
plt.tight_layout()
plt.savefig("f_loss_curve.png", dpi=120)
plt.show()


# %%
# 9.1.g Test the model and generate a confusion matrix
# ============================================================

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_pred_prob   = model.forward(X_test)
y_pred_labels = (y_pred_prob >= 0.5).astype(int).flatten()
y_test_flat   = y_test.flatten()

accuracy = np.mean(y_pred_labels == y_test_flat)
print(f"Test Accuracy: {accuracy * 100:.1f}%")

cm   = confusion_matrix(y_test_flat, y_pred_labels)
disp = ConfusionMatrixDisplay(cm, display_labels=[0, 1])

fig, ax = plt.subplots(figsize=(4, 4))
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Confusion Matrix – Test Set")
plt.tight_layout()
plt.savefig("g_confusion_matrix.png", dpi=120)
plt.show()


# %%
# 9.1.h Overfitting or Underfitting?
# ============================================================

# The training loss decreases steadily across 100 epochs and the test
# accuracy is high, which means the model generalises well.
#
# → The model shows no significant overfitting or underfitting.
#
# If underfitting occurred (high train loss, low accuracy):
#   - Train for more epochs
#   - Add more hidden neurons
#   - Increase the learning rate
#
# If overfitting occurred (low train loss, low test accuracy):
#   - Add dropout or L2 regularisation
#   - Reduce model complexity
#   - Use more training data


# %%
# 9.1.i Plot Ground Truth vs Predictions side by side
# ============================================================

colors = {0: "purple", 1: "yellow"}

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Ground Truth
for label in [0, 1]:
    mask = y_test_flat == label
    axes[0].scatter(X_test[mask, 0], X_test[mask, 1],
                    c=colors[label], edgecolors="grey",
                    linewidths=0.4, s=30, label=str(label))
axes[0].set_title("Ground Truth")
axes[0].set_xlabel("x_1")
axes[0].set_ylabel("x_2")
axes[0].legend(title="Class")

# Predictions
for label in [0, 1]:
    mask = y_pred_labels == label
    axes[1].scatter(X_test[mask, 0], X_test[mask, 1],
                    c=colors[label], edgecolors="grey",
                    linewidths=0.4, s=30, label=str(label))
axes[1].set_title("Predictions")
axes[1].set_xlabel("x_1")
axes[1].set_ylabel("x_2")
axes[1].legend(title="Class")

plt.tight_layout()
plt.savefig("i_ground_truth_vs_predictions.png", dpi=120)
plt.show()




