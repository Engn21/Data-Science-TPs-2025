

import warnings

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


warnings.filterwarnings("ignore", message=".*matmul", category=RuntimeWarning)


def select_with_label(images, labels, desired_labels):
    """Helper provided in the assignment for pulling specific digits."""
    mask = np.isin(labels, desired_labels)
    return images[mask], labels[mask]


# Sets a random seed so our sampling stays reproducible across reruns.
SEED = 42
rng = np.random.default_rng(SEED)

# Downloading MNIST exactly the way the handout suggests.
images, labels = fetch_openml(
    "mnist_784", return_X_y=True, as_frame=False, parser="auto"
)
train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, random_state=SEED
)

# The code focuses on digits 0–4 .
digit_subset = ["0", "1", "2", "3", "4"]
train_images_filtered, train_labels_filtered = select_with_label(
    train_images, train_labels, desired_labels=digit_subset
)
test_images_filtered, test_labels_filtered = select_with_label(
    test_images, test_labels, desired_labels=digit_subset
)

# Cponverting labels to ints and scale pixel values to [0, 1].
train_labels_filtered = train_labels_filtered.astype(np.int64)
test_labels_filtered = test_labels_filtered.astype(np.int64)
train_images_filtered = train_images_filtered.astype(np.float64) / 255.0
test_images_filtered = test_images_filtered.astype(np.float64) / 255.0

# The code takes exactly 1000 training samples per class so the set can stays balanced.
X_parts, y_parts = [], []
for digit in range(5):
    digit_idx = np.where(train_labels_filtered == digit)[0]
    chosen = rng.choice(digit_idx, size=1000, replace=False)
    X_parts.append(train_images_filtered[chosen])
    y_parts.append(train_labels_filtered[chosen])

X_train = np.vstack(X_parts)
y_train = np.concatenate(y_parts)

# Shuffle the balanced training set one more time just to mix classes.
perm = rng.permutation(len(y_train))
X_train = X_train[perm]
y_train = y_train[perm]

# Keep all filtered test samples for evaluation.
X_test = test_images_filtered
y_test = test_labels_filtered

#build a 1-NN baseline on the original pixels (no PCA situation).
baseline_knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
baseline_knn.fit(X_train, y_train)
y_pred_baseline = baseline_knn.predict(X_test)
baseline_acc = accuracy_score(y_test, y_pred_baseline) * 100
print(f"Baseline 1-NN accuracy (no PCA): {baseline_acc:.2f}%")

# List of PCA dimensions we are aiming  to try.
m_values = [10, 20, 30, 40, 50] + list(range(100, 751, 50))

accuracies_pca = []

# Here for each m, fit PCA, reconstruct data, then I run 1-NN on the reconstructions.
for m in m_values:
    print(f"Working on m = {m} principal components...")

    # fitting PCA on the training data only.
    pca = PCA(n_components=m, random_state=SEED, svd_solver="randomized")
    pca.fit(X_train)


    X_train_recon = pca.inverse_transform(pca.transform(X_train))
    X_test_recon = pca.inverse_transform(pca.transform(X_test))

    # Run 1-NN on the reconstructed images.
    knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
    knn.fit(X_train_recon, y_train)
    y_pred = knn.predict(X_test_recon)
    acc = accuracy_score(y_test, y_pred) * 100
    accuracies_pca.append(acc)

    print(f"    accuracy = {acc:.2f}%")

# Visualizes how accuracy changes for better understanding.
plt.figure(figsize=(10, 6))
plt.plot(m_values, accuracies_pca, marker="o", label="PCA + 1-NN (reconstructed)")
plt.axhline(
    y=baseline_acc,
    color="r",
    linestyle="--",
    label=f"Baseline (no PCA) = {baseline_acc:.2f}%",
)
plt.title("PCA + 1-NN accuracy vs. number of principal components")
plt.xlabel("Number of principal components (m)")
plt.ylabel("Classification accuracy (%)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


print("\nAccuracy table (m vs. PCA+1NN accuracy)")
for m, acc in zip(m_values, accuracies_pca):
    print(f"  m = {m:3d} → accuracy = {acc:.2f}%")
