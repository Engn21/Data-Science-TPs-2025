import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def get_twos(num=5000, seed=7):
    """
    The code here loads MNIST dataset, takes only digit '2' and returns up to 'num' samples
    everything here is just data prep, PCA is done by hand below
    """
    # download mnist (each image is 28x28=784 pixels)
    imgs, lbls = fetch_openml('mnist_784', return_X_y=True, as_frame=False, parser='auto')
    # splitting
    Xtr, _, ytr, _ = train_test_split(imgs, lbls, random_state=seed)

    # take only those rows where the label is '2'
    only2 = Xtr[ytr == '2']

    # choose randomly 'num' samples (or less if dataset smaller)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(only2), size=min(num, len(only2)), replace=False)

    # convert to float and scale pixel values to [0, 1]
    # this helps with numerical stability
    X = only2[idx].astype(float) / 255.0
    return X



def my_pca(data):
    """
    This class manually computes PCA using covariance matrix + eigen decomposition
    basically what sklearn.PCA does internally, but step by step
    """
    # center the data (remove mean from each column)
    # we subtract the mean of each feature (pixel) from all samples
    # this step is critical because PCA only cares about variance, not mean
    mu = np.mean(data, axis=0, keepdims=True)
    data_c = data - mu

    n = data_c.shape[0]  # number of samples

    #  compute the sample covariance matrix
    # (d x d) where d is number of features (784 for MNIST)
    # the covariance tells how features vary together
    cov = (data_c.T @ data_c) / (n - 1)

    # 3. find eigenvalues and eigenvectors of the covariance matrix
    # since cov is symmetric, we use eigh (more stable than eig)
    vals, vecs = np.linalg.eigh(cov)

    # eigh returns eigenvalues in ascending order → reverse it
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    # eigenvectors are the principal directions,
    # eigenvalues tell how much variance each direction explains
    return vals, vecs, mu.ravel()


def cum_var(lmbd):
    """
    computes cumulative explained variance ratio:
    S(k) = (sum_{j=1}^k λ_j) / (sum_{j=1}^d λ_j)
    this tells us how much of total variance is captured by the first k PCs
    """
    s = np.sum(lmbd)       # total variance
    cs = np.cumsum(lmbd)   # partial sums up to each k
    return cs / s          # ratio (0 → 1)

#  load 5000 samples of digit '2'
X = get_twos(5000, seed=42)

# perform PCA using only NumPy
eig_val, eig_vec, mean_vec = my_pca(X)

# compute cumulative explained variance
cev = cum_var(eig_val)

# plot the curve
ks = np.arange(1, len(cev) + 1)
plt.figure(figsize=(7,4.5))
plt.plot(ks, cev, lw=2)
plt.xlabel("Number of components (k)")
plt.ylabel("Cumulative explained variance")
plt.title("Cumulative explained variance vs. k (for MNIST digit '2')")
plt.ylim(0, 1.01)
plt.grid(alpha=0.3)

# draw dashed lines for 90% and 95% thresholds
for limit in [0.90, 0.95]:
    kidx = np.searchsorted(cev, limit) + 1
    plt.axhline(limit, ls="--", alpha=0.5)
    plt.axvline(kidx, ls="--", alpha=0.5)
    plt.text(kidx + 5, limit - 0.03, f"{int(limit*100)}% ≈ k={kidx}", fontsize=9)

plt.show()

# print some info
print("Total variance:", eig_val.sum())
print("Top 10 variance ratios:", np.round(100 * np.diff(np.hstack(([0.0], cev[:10]))), 2))
print("k for 90% variance:", np.searchsorted(cev, 0.90) + 1)
print("k for 95% variance:", np.searchsorted(cev, 0.95) + 1)
