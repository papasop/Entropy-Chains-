import numpy as np
from scipy.stats import pearsonr
from pyinform import transfer_entropy
from statsmodels.tsa.stattools import adfuller
import os
from mpmath import mp

# Disable frozen modules to avoid debugger warnings
os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'

# Set precision to 50 decimal places
mp.dps = 50

# Function to compute Riemann zeta zeros with high precision
def compute_riemann_zeros(n_zeros=200):
    zeros = []
    t = mp.mpf('10')
    for i in range(n_zeros):
        zero = mp.zetazero(i + 1)
        if zero.imag > t:
            zeros.append(float(zero.imag))
            t = zero.imag
    return np.array(zeros, dtype=np.float64)

# Compute 200 zeros with 50-digit precision
# Replace with LMFDB data[](https://www.lmfdb.org/zeros/zeta/) if available
try:
    gamma = compute_riemann_zeros(200)
except Exception as e:
    print(f"Error computing Riemann zeros: {e}")
    raise

# Data validation
if np.any(gamma <= 0):
    raise ValueError("Invalid gamma values: zeros must be positive")
if np.any(np.isnan(gamma)):
    raise ValueError("NaN detected in gamma values")

n = np.arange(1, len(gamma) + 1)

# Simplified chain computation
def compute_chain(factor=2.0, c=1.1):
    try:
        tau = (np.pi / 2) - np.arctan(factor * gamma)
        phi = (4 / (np.pi * n)) * (np.cos(np.log(gamma)) + c)
        if np.any(np.isnan(phi)):
            raise ValueError("NaN detected in phi calculation")
        return tau, phi
    except Exception as e:
        print(f"Error in compute_chain: {e}")
        return None, None

# Compute for factor=2, c=1.1
tau, phi = compute_chain(factor=2.0, c=1.1)
if tau is None:
    raise ValueError("Computation failed")

# Log transform φ(n) for stationarity
phi_log = np.log(np.abs(phi) + 1e-12)

# Stationarity test
def check_stationarity(series, name):
    try:
        result = adfuller(series)
        print(f"ADF Test for {name}: p-value = {result[1]:.6f}")
        return result[1] < 0.05
    except Exception as e:
        print(f"Error in ADF test for {name}: {e}")
        return False

tau_stationary = check_stationarity(tau, "τ_n")
phi_stationary = check_stationarity(phi_log, "φ(n) (log-transformed)")
if not (tau_stationary and phi_stationary):
    print("Warning: Non-stationary series detected. Using differenced data.")
    tau_diff = np.diff(tau)
    phi_diff = np.diff(phi_log)
else:
    tau_diff, phi_diff = tau, phi_log

# Simple Transfer Entropy calculation
try:
    tau_norm = (tau_diff - np.mean(tau_diff)) / np.std(tau_diff)
    phi_norm = (phi_diff - np.mean(phi_diff)) / np.std(phi_diff)
    
    print("Transfer Entropy between τ_n and φ(n) (log-transformed, differenced):")
    for bins in [10, 20]:
        print(f"  With {bins} quantile bins:")
        tau_quantiles = np.percentile(tau_norm, np.linspace(0, 100, bins + 1))
        phi_quantiles = np.percentile(phi_norm, np.linspace(0, 100, bins + 1))
        tau_discrete = np.digitize(tau_norm, tau_quantiles)
        phi_discrete = np.digitize(phi_norm, phi_quantiles)
        for k in [1, 2, 3]:
            te = transfer_entropy(tau_discrete, phi_discrete, k=k, local=False)
            print(f"    k={k}: {te:.6f}")
except Exception as e:
    print(f"Error in transfer_entropy: {e}")

# Basic statistics for validation
print(f"Mean of τ_n: {np.mean(tau):.6f}")
print(f"Pearson correlation between τ_n and φ(n): {pearsonr(tau, phi)[0]:.6f}")