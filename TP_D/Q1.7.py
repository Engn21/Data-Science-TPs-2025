import numpy as np
import matplotlib.pyplot as plt

# True parameters
a_true = 2.0
lambda_true = 0.7
n = 100
Nexp = 5000   # number of experiments

# Lists to store estimators
a_mom_list = []
lambda_mom_list = []
a_mle_list = []
lambda_mle_list = []

for _ in range(Nexp):
    # Generate data
    X = a_true + np.random.exponential(scale=1/lambda_true, size=n)
    
    # Empirical stats
    X_bar = np.mean(X)
    S_n = np.sqrt(np.mean((X - X_bar)**2))
    
    # MoM estimators
    lambda_mom = 1 / S_n
    a_mom = X_bar - S_n
    
    # MLE estimators
    a_mle = np.min(X)
    lambda_mle = n / np.sum(X - a_mle)
    
    # Store
    a_mom_list.append(a_mom)
    lambda_mom_list.append(lambda_mom)
    a_mle_list.append(a_mle)
    lambda_mle_list.append(lambda_mle)

# Convert to numpy arrays
a_mom_list = np.array(a_mom_list)
lambda_mom_list = np.array(lambda_mom_list)
a_mle_list = np.array(a_mle_list)
lambda_mle_list = np.array(lambda_mle_list)

# HISTOGRAMS
plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
plt.hist(a_mom_list, bins=40, alpha=0.5, label="MoM a_hat")
plt.hist(a_mle_list, bins=40, alpha=0.5, label="MLE a_hat")
plt.legend()
plt.title("Distribution of a estimators")

plt.subplot(1,2,2)
plt.hist(lambda_mom_list, bins=40, alpha=0.5, label="MoM lambda_hat")
plt.hist(lambda_mle_list, bins=40, alpha=0.5, label="MLE lambda_hat")
plt.legend()
plt.title("Distribution of lambda estimators")

plt.show()

# PRINT MEAN AND VARIANCE
print("Mean / Variance of a estimators")
print("MoM:", np.mean(a_mom_list), np.var(a_mom_list))
print("MLE:", np.mean(a_mle_list), np.var(a_mle_list))

print("\nMean / Variance of lambda estimators")
print("MoM:", np.mean(lambda_mom_list), np.var(lambda_mom_list))
print("MLE:", np.mean(lambda_mle_list), np.var(lambda_mle_list))
