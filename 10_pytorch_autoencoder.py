# %%
# author: Noureldin Elnems, noe2466@thi.de, Group: 7
#         Md Rasel, mdr3149@thi.de, Group: 7

# ============================================================
# Exercise 10 - Autoencoder
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# ============================================================
# (a) Load dataset and split into train/test subsets
# ============================================================

digits = load_digits()

X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    train_size=0.8,
    random_state=42,
    shuffle=True
)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


# ============================================================
# (b) Normalize features to [-1, 1]
# ============================================================

scaler = MinMaxScaler(feature_range=(-1, 1))

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)


# ============================================================
# (c) Autoencoder model definition
# ============================================================

class Autoencoder(nn.Module):

    def __init__(self, feat_dim, latent_dim):

        super(Autoencoder, self).__init__()

        self.conv_encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=2,
                padding=1
            ),
            nn.Sigmoid()
        )

        self.encoder_fc = nn.Sequential(
            nn.Linear(81, 64),
            nn.Sigmoid(),
            nn.Linear(64, latent_dim)
        )

        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.Sigmoid(),
            nn.Linear(64, feat_dim)
        )

        self.output_activation = nn.Tanh()

    def encode(self, x):

        x = x.reshape(-1, 1, 8, 8)
        x = self.conv_encoder(x)
        x = x.reshape(x.shape[0], -1)
        z = self.encoder_fc(x)

        return z

    def decode(self, z):

        x_rec = self.decoder_fc(z)
        x_rec = self.output_activation(x_rec)

        return x_rec

    def forward(self, x):

        z = self.encode(x)
        x_rec = self.decode(z)

        return x_rec


feat_dim = X_train_tensor.shape[1]
latent_dim = 2

model = Autoencoder(feat_dim=feat_dim, latent_dim=latent_dim)

print(model)


# ============================================================
# (d) Loss function
# ============================================================

criterion = nn.MSELoss()


# ============================================================
# (e) Optimizer
# ============================================================

optimizer = optim.SGD(model.parameters(), lr=0.1)


# ============================================================
# (f) Train the autoencoder
# ============================================================

train_dataset = TensorDataset(X_train_tensor, X_train_tensor, y_train_tensor)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

num_epochs = 20

loss_values = []
latent_space_history = []

for epoch in range(num_epochs):

    model.train()
    epoch_loss = 0.0

    for input_batch, target_batch, label_batch in train_loader:

        optimizer.zero_grad()

        reconstructed_batch = model(input_batch)

        loss = criterion(reconstructed_batch, target_batch)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    average_loss = epoch_loss / len(train_loader)
    loss_values.append(average_loss)

    model.eval()

    with torch.no_grad():
        latent_train_epoch = model.encode(X_train_tensor)

    latent_space_history.append(latent_train_epoch.detach().numpy())

    print(f"Epoch {epoch + 1:02d}/{num_epochs}, Loss = {average_loss:.6f}")


plt.figure(figsize=(7, 5))
plt.plot(range(1, num_epochs + 1), loss_values, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Average MSE Loss")
plt.title("Training Loss over 20 Epochs")
plt.grid(True)
plt.show()


# ============================================================
# (g) Plot latent space representation
# ============================================================

model.eval()

with torch.no_grad():
    latent_train_final = model.encode(X_train_tensor).detach().numpy()

plt.figure(figsize=(7, 6))

for digit in range(10):

    indices = y_train == digit

    plt.scatter(
        latent_train_final[indices, 0],
        latent_train_final[indices, 1],
        s=15,
        label=str(digit),
        alpha=0.8
    )

plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title("Latent Space Representation")
plt.legend()
plt.grid(True)
plt.show()

print("\nInterpretation of the latent space:")
print("Each point is one digit image compressed into two latent values.")
print("Digits close together have similar learned representations.")
print("Overlapping classes mean some digits look similar or the 2D latent space is too small.")


# ============================================================
# (h) Plot original and reconstructed images
# ============================================================

digits_to_show = [0, 4, 8]

model.eval()

with torch.no_grad():

    for digit in digits_to_show:

        index = np.where(y_test == digit)[0][0]

        original_flat = X_test_tensor[index].reshape(1, -1)
        reconstructed_flat = model(original_flat)

        original_img = original_flat.detach().numpy().reshape(8, 8)
        reconstructed_img = reconstructed_flat.detach().numpy().reshape(8, 8)

        fig, axes = plt.subplots(1, 2, figsize=(6, 3))

        axes[0].imshow(original_img, cmap="gray", vmin=-1, vmax=1)
        axes[0].set_title(f"Input Image: {digit}")
        axes[0].set_xlabel("$x_1$")
        axes[0].set_ylabel("$x_2$")

        axes[1].imshow(reconstructed_img, cmap="gray", vmin=-1, vmax=1)
        axes[1].set_title("Reconstructed Image")
        axes[1].set_xlabel("$x_1$")
        axes[1].set_ylabel("$x_2$")

        plt.tight_layout()
        plt.show()


