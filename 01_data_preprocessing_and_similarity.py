# %%
import pandas as pd

# 1.1.a


data = {
    "Patient ID": ["000345", "000124", "001758", "000994", "001233", "001145", "000222"],
    "Age": [45, 60, 22, 38, 36, 77, 65],
    "Height [cm]": [167, 181, 158, 185, 164, 190, 180],
    "Weight [kg]": [67, 78, 57, 90, 72, 75, 110]
}

df = pd.DataFrame(data)

print("Initial table:")
print(df)
print()


# 1.1.b


new_patient = pd.DataFrame({
    "Patient ID": ["001122"],
    "Age": [51],
    "Height [cm]": [177],
    "Weight [kg]": [81]
})

df = pd.concat([df, new_patient], ignore_index=True)

print("Table after adding new patient:")
print(df)
print()


# 1.1.c

def func_norm(v):
    """
    Normalize a pandas Series or array-like vector to the range [0, 1]
    using min-max normalization:
        v_norm = (v - v_min) / (v_max - v_min)
    """
    v_min = v.min()
    v_max = v.max()

    # Avoid division by zero if all values are the same
    if v_max == v_min:
        return v * 0

    return (v - v_min) / (v_max - v_min)


# 1.1.e


df["Age_norm"] = func_norm(df["Age"])
df["Height_norm"] = func_norm(df["Height [cm]"])
df["Weight_norm"] = func_norm(df["Weight [kg]"])

print("Table after adding normalized columns:")
print(df)
print()

# 1.1.f


df.to_csv("patients.csv", index=False)
print("Saved extended table to 'patients.csv'")
print()


# 1.1.g

average_age = df["Age"].mean()

youngest_df = df[df["Age"] <= average_age]

youngest_df.to_csv("youngest_patients.csv", index=False)

print(f"Average age = {average_age}")
print("Patients below or equal to average age:")
print(youngest_df)
print()
print("Saved to 'youngest_patients.csv'")

# %%

import matplotlib.pyplot as plt
import numpy as np

# 1.2.a

df = pd.read_csv("patients.csv")

print("Loaded data:")
print(df.head())
print()

# 1.2.b

heights = df["Height [cm]"]
num_patients = len(df)

plt.figure()
plt.hist(heights, bins=num_patients, edgecolor='black')
plt.xlabel("Height [cm]")
plt.ylabel("N")
plt.title("Histogram of Heights")

plt.show()

# 1.2.c

age = df["Age_norm"]
height = df["Height_norm"]
weight = df["Weight_norm"]

plt.figure(figsize=(10, 4))

# Age vs Height
plt.subplot(1, 2, 1)
plt.scatter(age, height)
plt.xlabel("Age")
plt.ylabel("Height")
plt.title("Age vs Height")

# Age vs Weight
plt.subplot(1, 2, 2)
plt.scatter(age, weight)
plt.xlabel("Age")
plt.ylabel("Weight")
plt.title("Age vs Weight")

plt.tight_layout()
plt.show()


# 1.2.d

def pearson_corr(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sqrt(
        np.sum((x - x_mean) ** 2) * np.sum((y - y_mean) ** 2)
    )

    return numerator / denominator


r_age_height = pearson_corr(age, height)
r_age_weight = pearson_corr(age, weight)

print("Pearson Correlations:")
print(f"Age vs Height: {r_age_height}")
print(f"Age vs Weight: {r_age_weight}")


