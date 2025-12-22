import numpy as np

# True parameters
a_true = 2.0
lambda_true = 0.7
n = 100

# 1. Generate sample
X = a_true + np.random.exponential(scale=1/lambda_true, size=n)

# 2. Empirical statistics
X_bar = np.mean(X)
S_n = np.sqrt(np.mean((X - X_bar)**2))

# 3. MoM estimators
lambda_mom = 1 / S_n
a_mom = X_bar - S_n

# 4. MLE estimators
a_mle = np.min(X)
lambda_mle = n / np.sum(X - a_mle)

print("MoM estimators:")
print("a_MoM =", a_mom)
print("lambda_MoM =", lambda_mom)

print("\nMLE estimators:")
print("a_MLE =", a_mle)
print("lambda_MLE =", lambda_mle)
