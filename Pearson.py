import numpy as np
import matplotlib.pyplot as plt
from mpmath import mp, zetazero, cos, log, pi
from scipy.interpolate import make_interp_spline
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")

# 设置高精度
mp.dps = 50

# --- 1. 获取前 N 个 Riemann 零点的虚部 γₙ ---
def get_gamma_n(N):
    return [mp.im(zetazero(n)) for n in range(1, N + 1)]

# --- 2. 构造 φ(n) ---
def phi(n, gamma_n):
    return (4 / (pi * n)) * (cos(log(gamma_n[n - 1])) + 1.1)

# --- 3. 结构熵 H(n) ---
def H(phi_n):
    return np.log(1 + phi_n**2)

# --- 4. K(n) = d log|φ| / d log H ---
def compute_K(phi_vals, H_vals):
    log_phi = np.log(np.abs(phi_vals))
    log_H = np.log(H_vals)
    d_log_phi = np.gradient(log_phi)
    d_log_H = np.gradient(log_H)
    return d_log_phi / d_log_H

# --- 5. 高阶导数 κ(n) = dK/dn ---
def compute_kappa(K_vals):
    return np.gradient(K_vals)

# --- 6. 几何相位 τₙ ---
def compute_tau(gamma_array):
    return np.pi / 2 - np.arctan(2 * gamma_array)

# --- 7. 拟合 φ̂(n) 用 B样条（模拟重建） ---
def fit_phi_spline(n_vals, phi_vals):
    spline = make_interp_spline(n_vals, phi_vals, k=3)
    return spline(n_vals), spline

# 主流程
N = 200
n_vals = np.arange(1, N + 1)
gamma_list = get_gamma_n(N)
gamma_array = np.array([float(g) for g in gamma_list])
phi_vals = np.array([float(phi(n + 1, gamma_list)) for n in range(N)])
H_vals = H(phi_vals)
K_vals = compute_K(phi_vals, H_vals)
kappa_vals = compute_kappa(K_vals)
tau_vals = compute_tau(gamma_array)

# 用 B 样条模拟重建 φ̂(n)
phi_hat_vals, spline_phi = fit_phi_spline(n_vals, phi_vals)
H_hat = H(phi_hat_vals)
K_hat = compute_K(phi_hat_vals, H_hat)
kappa_hat = compute_kappa(K_hat)
R_hat = phi_hat_vals / np.sqrt(H_hat)

# 输出核心统计
print(f"📌 φ̂ 与 φ 的 Pearson: {pearsonr(phi_vals, phi_hat_vals)[0]:.5f}")
print(f"✅ 平均 K(n) ≈ {np.mean(K_hat):.5f}")
print(f"σ(K(n)) ≈ {np.std(K_hat):.5f}")
print(f"✅ 平均 R(n) = φ/√H ≈ {np.mean(R_hat):.5f}")
print(f"σ(R(n)) ≈ {np.std(R_hat):.5f}")
print(f"✅ 平均 κ(n) ≈ {np.mean(kappa_hat):.5f}")
print(f"σ(κ(n)) ≈ {np.std(kappa_hat):.5f}")
print(f"τ_n 与 φ(n) 的 Pearson 相关系数: {pearsonr(tau_vals, phi_vals)[0]:.5f}")

# 可视化结构链条
plt.figure(figsize=(14, 5))
plt.subplot(1, 3, 1)
plt.plot(n_vals, phi_hat_vals, label='φ̂(n)', color='blue')
plt.title("φ̂(n) B-Spline 重建")
plt.grid(); plt.legend()

plt.subplot(1, 3, 2)
plt.plot(n_vals, K_hat, label='K(n)', color='green')
plt.axhline(0.5, color='gray', linestyle='--')
plt.title("结构比率 K(n)")
plt.grid(); plt.legend()

plt.subplot(1, 3, 3)
plt.plot(n_vals, kappa_hat, label='κ(n)', color='red')
plt.title("K(n) 的导数 κ(n)")
plt.grid(); plt.legend()

plt.tight_layout()
plt.show()
