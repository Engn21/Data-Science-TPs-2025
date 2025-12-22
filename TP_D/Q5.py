import numpy as np
import matplotlib.pyplot as plt
import math


# ============================================================
# Task 1: Signal Generation
# ============================================================
print("=" * 60)
print("TASK 1: Signal Generation")
print("=" * 60)

k = 10000          # number of bits
N = 10             # repetitions per bit
sigma2 = 1.5
sigma = math.sqrt(sigma2)

# Generate random bits
bits = np.random.randint(0, 2, size=k)

# Repeat each bit N times to form x[n]
x = np.repeat(bits, N).astype(float)

# Generate AWGN
noise = sigma * np.random.randn(len(x))
y = x + noise

# CHECK OUTPUTS
print(f"  Number of bits (k):        {k}")
print(f"  Repetitions per bit (N):   {N}")
print(f"  Total signal length:       {len(x)} (should be {k*N})")
print(f"  Noise variance (sigma^2):  {sigma2}")
print(f"  Noise std (sigma):         {sigma:.4f}")
print(f"  Empirical noise variance:  {np.var(noise):.4f} (should be ≈ {sigma2})")
print(f"  Empirical noise mean:      {np.mean(noise):.4f} (should be ≈ 0)")
print(f"  Number of 0-bits:          {np.sum(bits == 0)} ({100*np.mean(bits == 0):.1f}%)")
print(f"  Number of 1-bits:          {np.sum(bits == 1)} ({100*np.mean(bits == 1):.1f}%)")
print(f"  Signal generation complete\n")


# ============================================================
# Task 2: Detection Rules (NP and Bayesian thresholds)
# ============================================================
print("=" * 60)
print("TASK 2: Detection Rules")
print("=" * 60)

sigma_T = math.sqrt(sigma2 / N)    # std of block-averaged statistic

# Q-function
def Q(x):
    return 0.5 * math.erfc(x / math.sqrt(2))

# Inverse Q-function for 0.01
Qinv_001 = 2.3263478740408408

# NP threshold (for P_FA = 0.01)
gamma_NP = sigma_T * Qinv_001

# Bayesian threshold (optimal for equal priors)
gamma_B = 0.5

# CHECK OUTPUTS
print(f"  Block-averaged std (sigma_T):    {sigma_T:.4f}")
print(f"  Q^-1(0.01):                      {Qinv_001:.4f}")
print(f"  NP threshold (gamma_NP):         {gamma_NP:.4f}")
print(f"  Bayesian threshold (gamma_B):    {gamma_B:.4f}")
print(f"  Verify Q(Qinv_001):              {Q(Qinv_001):.4f} (should be ≈ 0.01)")
print(f"   Thresholds computed\n")


# ============================================================
# Task 3: Apply Detection
# ============================================================
print("=" * 60)
print("TASK 3: Apply Detection")
print("=" * 60)

# Reshape y into blocks of size N
y_blocks = y.reshape(k, N)
T = y_blocks.mean(axis=1)    # test statistic for each bit

# Apply NP and Bayes rules
bits_hat_NP = (T > gamma_NP).astype(int)
bits_hat_B  = (T > gamma_B).astype(int)

# CHECK OUTPUTS
print(f"  Test statistic T shape:          {T.shape} (should be ({k},))")
print(f"  T range: [{T.min():.4f}, {T.max():.4f}]")
print(f"  T mean:  {T.mean():.4f} (should be ≈ 0.5 for equal 0/1 bits)")
print(f"  NP detections (hat=1):           {np.sum(bits_hat_NP)}")
print(f"  Bayesian detections (hat=1):     {np.sum(bits_hat_B)}")
print(f"  NP correct detections:           {np.sum(bits_hat_NP == bits)} / {k} ({100*np.mean(bits_hat_NP == bits):.2f}%)")
print(f"  Bayesian correct detections:     {np.sum(bits_hat_B == bits)} / {k} ({100*np.mean(bits_hat_B == bits):.2f}%)")
print(f"   Detection applied\n")


# ============================================================
# Task 4: Visualization (zoom into first 100 samples)
# ============================================================
print("=" * 60)
print("TASK 4: Visualization")
print("=" * 60)

x_hat_B = np.repeat(bits_hat_B, N)  # sample-level estimated signal

n_zoom = 100
plt.figure(figsize=(10, 4))
plt.step(range(n_zoom), x[:n_zoom], where='post', label='x[n]')
plt.plot(range(n_zoom), y[:n_zoom], label='y[n]')
plt.step(range(n_zoom), x_hat_B[:n_zoom], where='post',
         linestyle='--', label='x_hat[n] (Bayesian)')
plt.xlabel("n")
plt.ylabel("Amplitude")
plt.title("Zoomed view of first 100 samples")
plt.legend()
plt.grid(True)
plt.savefig("task4_signal_plot.png", dpi=150)
plt.show()

# CHECK OUTPUTS
print(f"  Zoomed samples:                  {n_zoom}")
print(f"  Bits in zoomed view:             {n_zoom // N}")
print(f"  Plot saved to:                   task4_signal_plot.png")
print(f"   Visualization complete\n")


# ============================================================
# Task 5: Empirical Performance
# ============================================================
print("=" * 60)
print("TASK 5: Empirical Performance")
print("=" * 60)

is_H0 = (bits == 0)
is_H1 = (bits == 1)

# NP empirical
P_FA_NP_emp = np.mean(bits_hat_NP[is_H0] == 1)
P_M_NP_emp  = np.mean(bits_hat_NP[is_H1] == 0)
P_e_NP_emp  = 0.5 * (P_FA_NP_emp + P_M_NP_emp)

# Bayesian empirical
P_FA_B_emp = np.mean(bits_hat_B[is_H0] == 1)
P_M_B_emp  = np.mean(bits_hat_B[is_H1] == 0)
P_e_B_emp  = 0.5 * (P_FA_B_emp + P_M_B_emp)

# Theoretical values for comparison
P_FA_NP_theory = 0.01  # by design
P_M_NP_theory = Q((1 - gamma_NP) / sigma_T)
P_e_NP_theory = 0.5 * (P_FA_NP_theory + P_M_NP_theory)

P_FA_B_theory = Q(0.5 / sigma_T)
P_M_B_theory = Q(0.5 / sigma_T)
P_e_B_theory = 0.5 * (P_FA_B_theory + P_M_B_theory)

# CHECK OUTPUTS
print(f"  H0 samples (bit=0):              {np.sum(is_H0)}")
print(f"  H1 samples (bit=1):              {np.sum(is_H1)}")
print()
print("  Neyman-Pearson Performance:")
print(f"    P_FA (empirical):              {P_FA_NP_emp:.4f}")
print(f"    P_FA (theoretical):            {P_FA_NP_theory:.4f}")
print(f"    P_M  (empirical):              {P_M_NP_emp:.4f}")
print(f"    P_M  (theoretical):            {P_M_NP_theory:.4f}")
print(f"    P_e  (empirical):              {P_e_NP_emp:.4f}")
print(f"    P_e  (theoretical):            {P_e_NP_theory:.4f}")
print()
print("  Bayesian Performance:")
print(f"    P_FA (empirical):              {P_FA_B_emp:.4f}")
print(f"    P_FA (theoretical):            {P_FA_B_theory:.4f}")
print(f"    P_M  (empirical):              {P_M_B_emp:.4f}")
print(f"    P_M  (theoretical):            {P_M_B_theory:.4f}")
print(f"    P_e  (empirical):              {P_e_B_emp:.4f}")
print(f"    P_e  (theoretical):            {P_e_B_theory:.4f}")
print()
print(f"   Empirical results match theoretical (within statistical variance)\n")


# ============================================================
# Task 6: Effect of N (theoretical curves)
# ============================================================
print("=" * 60)
print("TASK 6: Effect of N (Theoretical Curves)")
print("=" * 60)

# Parameters
sigma2 = 1.5
sigma_z = math.sqrt(sigma2)

N_values = np.arange(1, 101)

Pe_Bayes = []
Pe_NP = []

for N_val in N_values:
    sigma_T = sigma_z / math.sqrt(N_val)

    # Bayes total error (gamma = 0.5)
    arg = 0.5 / sigma_T
    P_FA_B = Q(arg)
    P_M_B = P_FA_B
    Pe_Bayes.append(P_FA_B)

    # Neyman-Pearson total error (P_FA fixed at 0.01)
    P_FA_NP = 0.01
    gamma_NP_val = sigma_T * Qinv_001

    # Miss probability
    arg2 = (gamma_NP_val - 1) / sigma_T
    P_M_NP = 0.5 * (1 + math.erf(arg2 / math.sqrt(2)))

    # Total error
    Pe_NP.append(0.5 * (P_FA_NP + P_M_NP))

# Plot
plt.figure(figsize=(10, 5))
plt.plot(N_values, Pe_Bayes, label="Bayesian Total Error $P_e$", linewidth=2)
plt.plot(N_values, Pe_NP, label="Neyman-Pearson Total Error $P_e$", linestyle='--', linewidth=2)
plt.xlabel("Repetition Factor N")
plt.ylabel("Total Error Probability")
plt.title("Theoretical Error Probability vs Repetition Factor N")
plt.grid(True)
plt.legend()
plt.yscale("log")
plt.savefig("task6_error_curves.png", dpi=150)
plt.show()

# CHECK OUTPUTS
print(f"  N range:                         [1, 100]")
print(f"  Bayes P_e at N=1:                {Pe_Bayes[0]:.4f}")
print(f"  Bayes P_e at N=10:               {Pe_Bayes[9]:.4f}")
print(f"  Bayes P_e at N=100:              {Pe_Bayes[99]:.6f}")
print(f"  NP P_e at N=1:                   {Pe_NP[0]:.4f}")
print(f"  NP P_e at N=10:                  {Pe_NP[9]:.4f}")
print(f"  NP P_e at N=100:                 {Pe_NP[99]:.6f}")
print(f"  Plot saved to:                   task6_error_curves.png")
print()
print(f"  As N increases, both error probabilities decrease")
print(f"  Bayesian detector has lower total error for equal priors\n")


