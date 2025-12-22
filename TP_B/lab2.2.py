import warnings

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

warnings.filterwarnings("ignore", message=".*matmul", category=RuntimeWarning)

#Here we are going to perform PCA on the MNIST dataset, specifically focusing on 
#the digit '2'. We will compute the covariance matrix, 
# perform eigenvalue decomposition, and plot the sorted eigenvalues.
X_all, y_all = fetch_openml("mnist_784", return_X_y=True, as_frame=False, parser="auto")
X_only_twos = X_all[y_all == "2"]  # (N2, 784)

# We are selecting 5000 random samples of the digit '2' for PCA analysis.
rng = np.random.default_rng(42)
idx = rng.choice(len(X_only_twos), size=5000, replace=False)
X = X_only_twos[idx].astype(np.float64) / 255.0   # (5000, 784)

# The steps include centering the data, computing the covariance matrix,
mu = X.mean(axis=0, keepdims=True)
Xc = X - mu

# Build the covariance matrix (1/n scaling works fine for PCA demos).
n = Xc.shape[0]
Sigma = (Xc.T @ Xc) / n

#Here we are going to perform PCA on the MNIST dataset, specifically focusing on 
#the digit '2'. We will compute the covariance matrix, 
# perform eigenvalue decomposition, and plot the sorted eigenvalues.
eigvals, eigvecs = np.linalg.eigh(Sigma)    # ascending order
eigvals = eigvals[::-1]                     # here comes in descending order
# eigvecs = eigvecs[:, ::-1]                # if needed later the directions can be reversed too

# drawing
plt.figure(figsize=(7, 4))
plt.plot(np.arange(1, len(eigvals) + 1), eigvals, marker="o", linewidth=1)
plt.xlabel("Component index j")
plt.ylabel("Eigenvalue $\\lambda_j$ (variance captured)")
plt.title("Sorted eigenvalues of the covariance matrix for digit '2'")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
