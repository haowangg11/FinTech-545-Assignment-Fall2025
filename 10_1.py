import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ===== 读取协方差矩阵 ===== #
df = pd.read_csv("data/test5_2.csv")
Sigma = df.values.astype(float)
n = Sigma.shape[0]

# ===== 核心公式函数 ===== #

def portfolio_vol(w, Sigma):
    w = np.array(w, dtype=float)
    return np.sqrt(np.dot(w, np.dot(Sigma, w)))

def CSD(w, Sigma):
    w = np.array(w, dtype=float)
    denom = np.dot(w, np.dot(Sigma, w))
    return (w * np.dot(Sigma, w)) / denom

def objective(w, Sigma):
    w = np.array(w, dtype=float)
    csd = CSD(w, Sigma)
    avg_csd = np.mean(csd)
    return np.sum((csd - avg_csd)**2)

# ===== 优化求 Risk Parity ===== #

w0 = np.ones(n) / n

constraints = ({
    "type": "eq",
    "fun": lambda w: np.sum(w) - 1
})

bounds = [(0, 1)] * n

res = minimize(
    objective,
    w0,
    args=(Sigma,),
    method="SLSQP",
    bounds=bounds,
    constraints=constraints
)

w_rp = np.array(res.x, dtype=float)

print("✅ Risk Parity Weights:")
print(w_rp)

# ===== 输出 CSV ===== #

out = pd.DataFrame({"W": w_rp})
out.to_csv("data/testout10_1.csv", index=False)
print("\n✅ Saved to data/testout10_1.csv")
