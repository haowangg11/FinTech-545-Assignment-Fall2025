import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ===== 读取协方差矩阵 ===== #
df = pd.read_csv("data/test5_2.csv")
Sigma = df.values.astype(float)
n = Sigma.shape[0]

# ===== 风险预算 (X5 = half risk share) ===== #
risk_budget = np.array([1, 1, 1, 1, 0.5])

# ===== 核心公式 ===== #

def CSD(w, Sigma):
    w = np.array(w, dtype=float)
    denom = np.dot(w, np.dot(Sigma, w))
    return (w * np.dot(Sigma, w)) / denom

def objective(w, Sigma, risk_budget):
    csd = CSD(w, Sigma)
    csd_adj = csd / risk_budget   # ✅ 风险预算调整
    avg = np.mean(csd_adj)
    return np.sum((csd_adj - avg)**2)

# ===== 优化 Risk Parity with Risk Budget ===== #

w0 = np.ones(n) / n

constraints = ({
    "type": "eq",
    "fun": lambda w: np.sum(w) - 1
})

bounds = [(0, 1)] * n

res = minimize(
    objective,
    w0,
    args=(Sigma, risk_budget),
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={
        'maxiter': 10000,
        'ftol': 1e-15
    }
)

w_rp = np.array(res.x, dtype=float)

print("✅ Risk Parity Weights (Half Risk Share to X5):")
print(w_rp)

# ===== 输出 CSV ===== #
out = pd.DataFrame({"W": w_rp})
out.to_csv("data/testout10_2.csv", index=False)
print("\n✅ Saved to data/testout10_2.csv")
