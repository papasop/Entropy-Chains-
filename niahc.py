import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# === Step 1: Hardcoded γₙ 数据（前 100 个非平凡 Riemann 零点虚部）===
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
n = np.arange(1, len(gamma)+1)

# === Step 2: 定义结构链 ===
tau = (np.pi/2) - np.arctan(2 * gamma)
phi = (4 / (np.pi * n)) * (np.cos(np.log(gamma)) + 1.1)
H = np.log(1 + phi**2)
log_phi = np.log(np.abs(phi))
log_H = np.log(H)

# === Step 3: 一阶与二阶导数 ===
dlog_phi = np.gradient(log_phi, log_H)
K = dlog_phi
d2log_phi = np.gradient(dlog_phi, log_H)
kappa = d2log_phi
R = phi / np.sqrt(H)

# === Step 4: 平均与标准差输出 ===
print(f"✅ 平均 K(n): {np.mean(K):.6f} | σ = {np.std(K):.6f}")
print(f"✅ 平均 R(n): {np.mean(R):.6f} | σ = {np.std(R):.6f}")
print(f"✅ 平均 κ(n): {np.mean(kappa):.6f} | σ = {np.std(kappa):.6f}")
corr_tau_phi = pearsonr(tau, phi)[0]
print(f"📌 Pearson(τ, φ): {corr_tau_phi:.5f}")

# === Step 5: 反推验证 ===
phi_hat = np.exp(0.5 * np.log(H + 1e-12))
pearson_phi = pearsonr(phi_hat, np.abs(phi))[0]
print(f"🔁 Pearson(φ̂, φ): {pearson_phi:.5f}")

# === Step 6: 可视化 ===
plt.figure(figsize=(12,6))
plt.subplot(3,1,1)
plt.plot(n, K, label='K(n)')
plt.axhline(0.5, color='gray', linestyle='--')
plt.title("Structure Ratio K(n)")
plt.grid()

plt.subplot(3,1,2)
plt.plot(n, R, label='R(n) = φ / √H')
plt.axhline(1, color='gray', linestyle='--')
plt.title("Amplitude-Entropy Ratio R(n)")
plt.grid()

plt.subplot(3,1,3)
plt.plot(n, kappa, label='κ(n)')
plt.axhline(0, color='gray', linestyle='--')
plt.title("Curvature κ(n)")
plt.grid()

plt.tight_layout()
plt.show()
