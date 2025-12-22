import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

# The code here grab digit "2" images from MNIST so we have a concrete toy set.
X_all, y_all = fetch_openml("mnist_784", return_X_y=True, as_frame=False, parser="auto")
X_all = X_all.astype(np.float64)
X = X_all[y_all == "2"][:5000]  # first 5k twosfor this demo

# Then we compute the average digit and center everything around it.
mean_vector = np.mean(X, axis=0)       # 784-dimensional mean image
X_centered = X - mean_vector           # subtract mean so PCA focuses on variance

# Build the covariance matrix (this is literally the core of PCA).
C = np.dot(X_centered.T, X_centered) / X_centered.shape[0]  # (784 x 784)

# Run eigen decomposition on the covariance matrix.
eigvals, eigvecs = np.linalg.eigh(C)   # eigh rocks for symmetric matrices like this

# Sort eigenvalues also eigenvectors from biggest to smallest.
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

# We are keeping the top 10 principal components for plotting.
pcs = eigvecs[:, :10].T   # (10, 784)

# Visualizing the mean digit and each of those PCs.
def plot_mean_and_pcs(mean_vector, pcs, k=10):
    plt.figure(figsize=(12, 5))
    # 1 mean + 10 PCs = 11 subplots total.
    for i in range(k + 1):
        plt.subplot(2, 6, i + 1)
        if i == 0:
            img = mean_vector.reshape(28, 28)
            plt.imshow(img, cmap="gray")
            plt.title("Mean")
        else:
            pc = pcs[i - 1].reshape(28, 28)
            vmax = np.abs(pc).max()
            plt.imshow(pc, cmap="bwr", vmin=-vmax, vmax=vmax)
            plt.title(f"PC{i}")
        plt.axis("off")
    plt.suptitle("Mean and first 10 Principal Components")
    plt.tight_layout()
    plt.show()


plot_mean_and_pcs(mean_vector, pcs, k=10)
