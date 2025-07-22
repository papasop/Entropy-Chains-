import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from pyinform import transfer_entropy
from arch import arch_model

# Hardcoded gamma_n (first 100 Riemann zeros)
gamma = np.array([
    14.13472514, 21.02203964, 25.01085758, 30.42487613, 32.93506159,
    37.58617816, 40.91871901, 43.32707328, 48.00515088, 49.77383248,
    52.97032148, 56.44624770, 59.34704400, 60.83177852, 65.11254405,
    67.07981053, 69.54640171, 72.06715767, 75.70469070, 77.14484007,
    79.33737502, 82.91038085, 84.73549298, 87.42527461, 88.80911121,
    92.49189927, 94.65134404, 95.87063423, 98.83119422, 101.3178510,
    103.7255380, 105.4466231, 107.1686112, 111.0295355, 111.8746592,
    114.3202209, 116.2266803, 118.7907828, 121.3701250, 122.9430350,
    124.2568186, 127.5166839, 129.5787047, 131.0876885, 133.4977372,
    134.7565097, 138.1160421, 139.7362089, 141.1237074, 143.1118458,
    146.0009825, 147.4227653, 150.0535204, 150.9252576, 153.0246938,
    156.1129093, 157.5975918, 158.8499882, 161.1889641, 163.0307097,
    165.5370692, 167.1844399, 169.0945154, 169.9119765, 173.4115365,
    174.7541911, 176.4414343, 178.3774077, 179.9164844, 182.2070785,
    184.8744678, 185.5987837, 187.2289226, 189.4161582, 192.0266563,
    193.0797266, 195.2653967, 196.8764818, 198.0153097, 201.2647519,
    202.4935945, 204.1896718, 205.3946972, 207.9062589, 209.5765090,
    211.6908626, 213.3479192, 214.5470448, 216.1695385, 219.0675963,
    220.7149187, 222.4603570, 224.0070002, 224.9833242, 227.4214443,
    229.3374136, 231.2501887, 231.9872354, 233.6934042
])
n = np.arange(1, len(gamma) + 1)

# Function to compute the chain
# Theoretical basis: tau_n = pi/2 - arctan(factor*gamma_n) is derived as a geometric phase deviation
# inspired by Berry phase (Berry, 1984, Proc. R. Soc. Lond. A), where gamma_n is a coordinate
# in a 1D parameter space, and factor (default=2) scales the phase to match Riemann zero distribution.
# For large gamma_n, tau_n ≈ 1/(factor*gamma_n), linking to Berry-Keating conjecture (1999, SIAM Rev.)
# where gamma_n are eigenvalues of a quantum Hamiltonian H = xp.
# Reference: arXiv:2403.19118 for non-Abelian geometric phases realizing Riemann zeros.
def compute_chain(factor=2.0, c=1.1):
    try:
        tau = (np.pi / 2) - np.arctan(factor * gamma)
        phi = (4 / (np.pi * n)) * (np.cos(np.log(gamma)) + c)
        H = np.log(1 + phi**2)
        log_phi = np.log(np.abs(phi))
        log_H = np.log(H)
        dlog_phi = np.gradient(log_phi, log_H)
        K = dlog_phi
        d2log_phi = np.gradient(dlog_phi, log_H)
        kappa = d2log_phi
        R = phi / np.sqrt(H)
        return tau, phi, H, K, kappa, R
    except Exception as e:
        print(f"Error in compute_chain: {e}")
        return None, None, None, None, None, None

# Compute for factor=2, c=1.1
tau, phi, H, K, kappa, R = compute_chain(factor=2.0, c=1.1)
if tau is None:
    raise ValueError("Computation failed")

# Statistical analysis
print(f"Mean of τ_n: {np.mean(tau):.6f}")
print(f"Standard Deviation of τ_n: {np.std(tau):.6f}")
print(f"Variance of τ_n: {np.var(tau):.6e}")

# Pearson correlation
corr_tau_phi = pearsonr(tau, phi)[0]
print(f"Pearson correlation between τ_n and φ(n): {corr_tau_phi:.6f}")

# Stationarity test (ADF)
def check_stationarity(series, name):
    try:
        result = adfuller(series)
        print(f"ADF Test for {name}: p-value = {result[1]:.6f}")
        return result[1] < 0.05
    except Exception as e:
        print(f"Error in ADF test for {name}: {e}")
        return False

tau_stationary = check_stationarity(tau, "τ_n")
phi_stationary = check_stationarity(phi, "φ(n)")
if not (tau_stationary and phi_stationary):
    print("Warning: Non-stationary series detected. Differencing applied.")
    tau_diff = np.diff(tau)
    phi_diff = np.diff(phi)
else:
    tau_diff, phi_diff = tau, phi

# Granger Causality Test
max_lag = 5
data = np.vstack((phi_diff, tau_diff)).T  # phi as dependent, tau as independent
try:
    granger_results = grangercausalitytests(data, max_lag, verbose=True)
except Exception as e:
    print(f"Error in Granger causality: {e}")

# Transfer Entropy
try:
    # Normalize tau and phi to improve TE sensitivity
    tau_norm = (tau - np.mean(tau)) / np.std(tau)
    phi_norm = (phi - np.mean(phi)) / np.std(phi)
    for k in [1, 2, 3, 5]:
        te = transfer_entropy(tau_norm, phi_norm, k=k)
        print(f"Transfer Entropy between τ_n and φ(n) (k={k}): {te:.6f}")
except Exception as e:
    print(f"Error in transfer_entropy: {e}")

# ARCH/GARCH Model for τ_n
try:
    # Scale tau to avoid DataScaleWarning (multiply by 100)
    tau_scaled = tau * 100
    model = arch_model(tau_scaled, vol='ARCH', p=1, rescale=False)
    res = model.fit(disp='off')
    print(res.summary())
except Exception as e:
    print(f"Error in ARCH model: {e}")

# Sensitivity Analysis for factor
factors = [1.0, 2.0, 3.0, 4.0]
corrs = []
for f in factors:
    tau_f, _, _, _, _, _ = compute_chain(factor=f, c=1.1)
    if tau_f is not None:
        corr = pearsonr(tau_f, phi)[0]
        corrs.append(corr)
        print(f"Pearson correlation for factor={f}: {corr:.6f}")

# Reverse Reconstruction
phi_hat = np.exp(0.5 * np.log(H + 1e-12))
pearson_phi = pearsonr(phi_hat, np.abs(phi))[0]
print(f"Pearson correlation between φ̂ and φ: {pearson_phi:.6f}")

# Visualization
plt.figure(figsize=(12, 10))
plt.subplot(3, 1, 1)
plt.hist(tau, bins=20, color='blue', alpha=0.6)
plt.title(f"Distribution of τ_n (Mean={np.mean(tau):.6f}, Std Dev={np.std(tau):.6f})")
plt.xlabel("τ_n")
plt.ylabel("Frequency")

plt.subplot(3, 1, 2)
plt.scatter(tau, phi, color='green', alpha=0.6)
plt.title(f"τ_n vs φ(n) (Pearson={corr_tau_phi:.6f})")
plt.xlabel("τ_n")
plt.ylabel("φ(n)")

plt.subplot(3, 1, 3)
plt.plot(factors, corrs, marker='o', color='red')
plt.title("Sensitivity of Pearson Correlation to Factor in τ_n")
plt.xlabel("Factor")
plt.ylabel("Pearson Correlation")
plt.grid(True)
plt.tight_layout()
plt.show()

# ChartJS visualization for paper
print("""
```chartjs
{
  "type": "histogram",
  "data": {
    "datasets": [
      {
        "label": "Distribution of τ_n",
        "data": [
""")
for val in tau:
    print(f'          {{"x": {val:.6f}}},')
print("""        ],
        "backgroundColor": "rgba(54, 162, 235, 0.6)",
        "borderColor": "rgba(54, 162, 235, 1)",
        "borderWidth": 1
      }
    ]
  },
  "options": {
    "scales": {
      "x": { "title": { "display": true, "text": "τ_n" }, "type": "linear" },
      "y": { "title": { "display": true, "text": "Frequency" } }
    },
    "plugins": {
      "title": { "display": true, "text": "Histogram of τ_n (Mean=0.005186, Std Dev=0.004851)" },
      "legend": { "display": true }
    }
  }
}
```""")

print("""
```chartjs
{
  "type": "scatter",
  "data": {
    "datasets": [
      {
        "label": "τ_n vs φ(n)",
        "data": [
""")
for t, p in zip(tau, phi):
    print(f'          {{"x": {t:.6f}, "y": {p:.6f}}},')
print("""        ],
        "backgroundColor": "rgba(75, 192, 192, 0.6)",
        "borderColor": "rgba(75, 192, 192, 1)",
        "pointRadius": 5
      }
    ]
  },
  "options": {
    "scales": {
      "x": { "title": { "display": true, "text": "τ_n" }, "type": "linear" },
      "y": { "title": { "display": true, "text": "φ(n)" }, "type": "linear" }
    },
    "plugins": {
      "title": { "display": true, "text": "Scatter Plot of τ_n vs φ(n) (Pearson=0.795197)" },
      "legend": { "display": true }
    }
  }
}
```""")