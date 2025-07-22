# Geometric Entropy Chains in the Riemann Spectrum

**Author**: Y.Y.N. Li  
**Date**: July 22, 2025  
**License**: MIT

## 📖 Overview

This project demonstrates a **causal entropy chain** rooted in the geometric phase deviations of the nontrivial zeros of the Riemann zeta function. We define a cascade of quantities:

where each structure arises naturally from the previous, showing strong statistical and functional coherence. The system reveals an **entropy attractor** with fixed structure ratio `K ≈ 0.5`, implying a hidden conservation law governing the Riemann spectrum.

---

## 🧠 Core Concepts

- **Riemann Zeros**: All zeros $\rho_n = \frac{1}{2} + i\gamma_n$ lie on the critical line (assuming RH).
- **Geometric Phase Deviation**:  
  $\tau_n = \frac{\pi}{2} - \arctan(2\gamma_n)$
- **Structural Amplitude**:  
  $\phi(n) = \frac{4}{\pi n} \left[\cos(\log \gamma_n) + c\right]$, with $c ≈ 1.1$
- **Entropy**:  
  $H(n) = \log(1 + \phi(n)^2)$
- **Structure Ratio**:  
  $K(n) = \frac{d \log |\phi(n)|}{d \log H(n)}$
- **Second-Order Curvature**:  
  $\kappa(n) = \frac{d^2 \log |\phi(n)|}{d \log^2 H(n)}$

---

## 📊 Statistical Highlights

- ✅ Average $K(n) ≈ 0.50026$, with std. dev. ≈ 0.00057  
- ✅ $\phi(n)/\sqrt{H(n)} ≈ 1.00030$, indicating an isometric link  
- ✅ Correlation between $\tau_n$ and $\phi(n)$: **0.84167**  
- ✅ Reverse-engineered $\hat{\phi}(n)$ (from fixed $K = 0.5$) has Pearson **1.00000** with original $\phi(n)$

---

## 🔁 Reversibility and Entropy Attractor

The project verifies **reversibility** in the entropy chain:  
Fixing $K(n) = 0.5$ allows accurate recovery of $\phi(n)$ from $H(n)$.  
This strongly suggests a **stable entropy structure attractor** governed by minimal phase deviation.

---

## 🔗 Files

- `main.ipynb` – Core analysis notebook (Python/NumPy/Matplotlib)
- `zeros.txt` – First 200 imaginary parts $\gamma_n$ of Riemann zeros
- `figs/` – Generated figures for entropy chains and statistical plots
- `paper.pdf` – Full paper (LaTeX compiled)  
- `README.md` – Project overview

---

## 🧮 Requirements

- Python ≥ 3.8  
- NumPy  
- Matplotlib  
- SciPy (optional)

Install via:

```bash
pip install numpy matplotlib scipy
