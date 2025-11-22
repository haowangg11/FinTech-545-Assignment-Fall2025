import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ===== Read Covariance Matrix ===== #
df = pd.read_csv("data/test5_3.csv")
Sigma = df.values.astype(float)

# ===== Read Means ===== #
means = pd.read_csv("data/test10_3_means.csv")["Mean"].values.astype(float)

rf = 0.04
n = len(means)

# ===== Sharpe Ratio Functions ===== #
def portfolio_vol(w, Sigma):
    return np.sqrt(np.dot(w, np.dot(Sigma, w)))

def portfolio_return(w, means):
    return np.dot(w, means)

def neg_sharpe(w, means, Sigma, rf):
    ret = portfolio_return(w, means)
    vol = portfolio_vol(w, Sigma)
    return -(ret - rf) / vol

# ===== Constraints and Bounds ===== #
w0 = np.ones(n) / n
constraints = ({
    "type": "eq",
    "fun": lambda w: np.sum(w) - 1
})
bounds = [(0, 1)] * n   # ✅ w >= 0

# ===== Optimization ===== #
res = minimize(
    neg_sharpe,
    w0,
    args=(means, Sigma, rf),
    method="SLSQP",
    bounds=bounds,
    constraints=constraints
)

w_msr = res.x

print("✅ Max Sharpe Weights (w >= 0):")
print(w_msr)

# ===== Output CSV ===== #
pd.DataFrame({"W": w_msr}).to_csv("data/testout10_3.csv", index=False)
print("\n✅ Saved to data/testout10_3.csv")
