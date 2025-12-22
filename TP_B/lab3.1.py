import warnings

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Sklearn 1.6 on macOS spams a harmless matmul RuntimeWarning, so I'm muting it.
warnings.filterwarnings("ignore", message=".*matmul", category=RuntimeWarning)


def select_with_label(images, labels, desired_labels):
    mask = np.isin(labels, desired_labels)
    return images[mask], labels[mask]


# lock in the seed so every rerun gives the same shuffled samples.
seed = 42
rng = np.random.default_rng(seed)
print("Using random seed:", seed)

# Pull MNIST from OpenML and split into train/test just like the handout says.
images, labels = fetch_openml(
    "mnist_784", return_X_y=True, as_frame=False, parser="auto"
)
train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, random_state=seed
)

#Trim the dataset so we only keep digits 0 through 4.
target_digits = [str(d) for d in range(5)]
train_images_slice, train_labels_slice = select_with_label(
    train_images, train_labels, desired_labels=target_digits
)
test_images_slice, test_labels_slice = select_with_label(
    test_images, test_labels, desired_labels=target_digits
)

# Turn labels into ints and scale pixel values into [0, 1].
train_labels_slice = train_labels_slice.astype(np.int64)
test_labels_slice = test_labels_slice.astype(np.int64)
train_images_slice = train_images_slice.astype(np.float64) / 255.0
test_images_slice = test_images_slice.astype(np.float64) / 255.0

# Here the code grabs exactly 1000 samples per digit so the training set stays balanced.
X_balanced, y_balanced = [], []
for digit in range(5):
    digit_indices = np.where(train_labels_slice == digit)[0]
    chosen = rng.choice(digit_indices, size=1000, replace=False)
    X_balanced.append(train_images_slice[chosen])
    y_balanced.append(train_labels_slice[chosen])
X_train_bal = np.vstack(X_balanced)
y_train_bal = np.concatenate(y_balanced)

# Keep the full test split (filtered to digits 0-4) for evaluation.
X_test = test_images_slice
y_test = test_labels_slice

# Next, hit the balanced set with PCA to squash it down to 2D for plotting.
pca = PCA(n_components=2, random_state=seed, svd_solver="full")
X_train_pca = pca.fit_transform(X_train_bal)
X_test_pca = pca.transform(X_test)

#Then the code standardizes the PCA features so distance-based models behave nicely.
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train_pca)
X_test_std = scaler.transform(X_test_pca)

# Pick a test sample and stash the true label for reference.
sample_index = 123
x_test = X_test_pca[sample_index]
x_test_std = X_test_std[sample_index]
true_label = y_test[sample_index]

k = 5

# Train vanilla k-NN on both raw PCA coords and the standardized version.
model_raw = KNeighborsClassifier(n_neighbors=k)
model_std = KNeighborsClassifier(n_neighbors=k)
model_raw.fit(X_train_pca, y_train_bal)
model_std.fit(X_train_std, y_train_bal)

# Pull the neighbor indices/distances for the chosen test point.
dist_raw, idx_raw = model_raw.kneighbors([x_test])
dist_std, idx_std = model_std.kneighbors([x_test_std])

# Compare predictions.
pred_raw = model_raw.predict([x_test])[0]
pred_std = model_std.predict([x_test_std])[0]

print(f"Actual label: {true_label}")
print(f"Prediction without standardization: {pred_raw}")
print(f"Prediction after standardization: {pred_std}")

# Visualize how the neighbors shift when we scale the features.
plt.figure(figsize=(12, 5))

# Plain PCA space.
plt.subplot(1, 2, 1)
plt.title("Raw PCA Space")
for digit in range(5):
    plt.scatter(
        X_train_pca[y_train_bal == digit, 0],
        X_train_pca[y_train_bal == digit, 1],
        label=str(digit),
        alpha=0.5,
    )
plt.scatter(
    x_test[0], x_test[1], color="green", s=150, label="Test Sample", edgecolor="black"
)
plt.scatter(
    X_train_pca[idx_raw[0], 0],
    X_train_pca[idx_raw[0], 1],
    facecolors="none",
    edgecolors="black",
    s=150,
    label="Neighbors",
)
plt.legend()

# Standardized PCA space.
plt.subplot(1, 2, 2)
plt.title("Standardized PCA Space")
for digit in range(5):
    plt.scatter(
        X_train_std[y_train_bal == digit, 0],
        X_train_std[y_train_bal == digit, 1],
        label=str(digit),
        alpha=0.5,
    )
plt.scatter(
    x_test_std[0],
    x_test_std[1],
    color="green",
    s=150,
    label="Test Sample",
    edgecolor="black",
)
plt.scatter(
    X_train_std[idx_std[0], 0],
    X_train_std[idx_std[0], 1],
    facecolors="none",
    edgecolors="black",
    s=150,
    label="Neighbors",
)
plt.legend()

plt.tight_layout()
plt.show()
