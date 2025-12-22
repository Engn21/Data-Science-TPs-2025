import warnings

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", message=".*matmul", category=RuntimeWarning)

# Download MNIST dataset
X_all, y_all = fetch_openml("mnist_784", return_X_y=True, as_frame=False, parser="auto")
X_all = X_all.astype(np.float64)

# The code selects 5000 samples of the digit '2' for PCA analysis
mask = (y_all == "2")
X_twos = X_all[mask]
rng = np.random.default_rng(42)
idx = rng.permutation(X_twos.shape[0])[:5000]
X = X_twos[idx]                       # (5000, 784)

# Average and centering the data
mu = X.mean(axis=0, keepdims=True)
Xc = X - mu

# PCA has been performed using sklearn's PCA
pca = PCA(n_components=784, svd_solver="randomized", random_state=42)
pca.fit(Xc)

# Eigenvalues and total variance
lambdas = pca.explained_variance_               # λ_j
total_var = lambdas.sum()

# Cumulative explained variance (CEV), tail, err(X,m), RE(m)
cev = np.cumsum(lambdas) / total_var            # length 784, cev[m] = ∑_{j=1}^{m} λ_j / ∑_{j=1}^{784} λ_j
tail = np.append(np.cumsum(lambdas[::-1])[::-1], 0.0)  # length 785, tail[m] = ∑_{j=m+1}^{784} λ_j
err = tail.copy()                                # err[m] = ∑_{j=m+1}^{784} λ_j 
RE  = err / err[0]                               # relative error

# 7) Thresholds: RE ≤ 0.50 ↔ CEV ≥ 0.50; RE ≤ 0.05 ↔ CEV ≥ 0.95; RE ≤ 0.01 ↔ CEV ≥ 0.99
def smallest_m_for_cev(th):
    i = np.searchsorted(cev, th, side="left")   # cev[i] >= th of i
    return int(i + 1) if i < len(cev) else 784

m_50 = smallest_m_for_cev(0.50)
m_95 = smallest_m_for_cev(0.95)
m_99 = smallest_m_for_cev(0.99)

print(f"RE(m) ≤ 0.50  → m = {m_50}")
print(f"RE(m) ≤ 0.05  → m = {m_95}")
print(f"RE(m) ≤ 0.01  → m = {m_99}")

# Plotting
plt.figure(figsize=(6, 4))
plt.plot(np.arange(0, len(err)), err)
plt.xlabel("m (number of components used)"); plt.ylabel("err(X,m)")
plt.title("Rebuilding error err(X,m)"); plt.tight_layout(); plt.show()

plt.figure(figsize=(6, 4))
plt.plot(np.arange(0, len(RE)), RE)
plt.xlabel("m"); plt.ylabel("RE(m) = err(X,m)/err(X,0)")
plt.title("Relative error RE(m)"); plt.tight_layout(); plt.show()
