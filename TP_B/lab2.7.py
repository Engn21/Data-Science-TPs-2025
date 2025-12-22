
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

# Loading MNIST from OpenML and peel out the digit "2" images.
X_all, y_all = fetch_openml("mnist_784", return_X_y=True, as_frame=False, parser="auto")
X_all = X_all.astype(np.float64)

X_digit2 = X_all[y_all == "2"]

# Splitting into a 5k training slice (for PCA) and a small held-out test block.
X_train = X_digit2[:5000]
X_test = X_digit2[5000:5050]

# Compute the mean digit and center training samples around it.
mean_vec = np.mean(X_train, axis=0)
X_centered = X_train - mean_vec

# Here in the code building the covariance matrix and run eigen decomposition.
C = np.dot(X_centered.T, X_centered) / X_centered.shape[0]

eigvals, eigvecs = np.linalg.eigh(C)  # eigh is built for symmetric matrices
idx = np.argsort(eigvals)[::-1]       # sort eigenvalues from largest to smallest
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

#H elper to reconstruct an image with the first m principal components.
def reconstruct_image(I, mean_vec, pcs, m):
    """
    Reconstruct image I using the first m principal components.
    I: (784,)
    mean_vec: (784,)
    pcs: (784, k)
    m: number of components to use
    """
    I_centered = I - mean_vec
    W = pcs[:, :m]
    coeffs = np.dot(W.T, I_centered)
    recon = mean_vec + np.dot(W, coeffs)
    return recon

# Sample five test images and rebuild them with m = 1..10.
np.random.seed(42)
num_imgs = 5
M = 10  # number of components to test
idx = np.random.choice(X_test.shape[0], num_imgs, replace=False)
images = X_test[idx]

pcs = eigvecs  # principal components

recons = []
losses = []

for img in images:
    img_recons = []
    img_losses = []
    for m in range(1, M + 1):
        r = reconstruct_image(img, mean_vec, pcs, m)
        loss = np.sum((img - r) ** 2)
        img_recons.append(r)
        img_losses.append(round(loss, 2))
    recons.append(img_recons)
    losses.append(img_losses)

# Visualize originals vs. reconstructions across m.
fig, axes = plt.subplots(num_imgs, M + 1, figsize=(15, 8))
fig.suptitle("PCA Reconstruction of 5 Test Images (NumPy PCA)")

for i in range(num_imgs):
    # Original image
    axes[i, 0].imshow(images[i].reshape(28, 28), cmap="gray")
    axes[i, 0].set_title("Original")
    axes[i, 0].axis("off")

    # Reconstructions for m = 1..10
    for m in range(M):
        axes[i, m + 1].imshow(recons[i][m].reshape(28, 28), cmap="gray")
        axes[i, m + 1].set_title(f"m={m + 1}\nloss={losses[i][m]}")
        axes[i, m + 1].axis("off")

plt.tight_layout()
plt.show()

# Here the code prints reconstruction losses for each image and m.
for i in range(num_imgs):
    print(f"Image {i+1} losses:")
    for m in range(M):
        print(f"  m={m+1:2d} -> loss={losses[i][m]}")
    print("-" * 30)
