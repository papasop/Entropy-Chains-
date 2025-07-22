import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# 假设我们已有的数据：
tau_n_all = np.array([0.001, 0.003, 0.005, 0.007, 0.009])
phi_n_all = 4 / (np.pi * np.arange(1, len(tau_n_all) + 1)) * (np.cos(np.log(np.arange(1, len(tau_n_all) + 1))) + 1.1)
H_n_all = np.log(1 + phi_n_all**2)
K_n_all = np.gradient(H_n_all, tau_n_all)  # 简化导数计算

# 使用二次多项式拟合：K_n ≈ a * tau^2 + b * tau + c
coeffs = np.polyfit(tau_n_all, K_n_all, deg=2)
a, b, c = coeffs
print("Fitted coefficients: a =", a, ", b =", b, ", c =", c)

# 可视化拟合效果
tau_range = np.linspace(0, 0.01, 100)
K_fit = a * tau_range**2 + b * tau_range + c

plt.scatter(tau_n_all, K_n_all, label='Data', color='red')
plt.plot(tau_range, K_fit, label='Quadratic Fit', color='blue')
plt.xlabel('τₙ')
plt.ylabel('Kₙ')
plt.legend()
plt.title('Quadratic Fit of Kₙ vs τₙ')
plt.grid(True)
plt.show()

# -------------------------
# 从目标 Kₙ 倒推 τₙ
# -------------------------

def inverse_tau(K_target):
    # 构造损失函数：误差平方
    def loss(tau):
        tau = tau[0]
        K_pred = a * tau**2 + b * tau + c
        return (K_pred - K_target)**2

    res = minimize(loss, x0=[0.005], bounds=[(0, 0.02)])
    return res.x[0]

# 目标 Kₙ 示例
K_target = 2.0
tau_reconstructed = inverse_tau(K_target)

print(f"Reconstructed τₙ from Kₙ = {K_target} → τₙ = {tau_reconstructed:.6f}")
