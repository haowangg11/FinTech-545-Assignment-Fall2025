import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ============================
# 1. 读入 in-sample / out-sample
# ============================
ins = pd.read_csv("data/problem3_insample.csv")
outs = pd.read_csv("data/problem3_outsample.csv")

# 去掉 Date，只保留 5 个资产的收益
R_in = ins.drop(columns=["Date"]).values   # (T_in, 5)
R_out = outs.drop(columns=["Date"]).values # (T_out, 5)
tickers = ins.columns[1:]                  # 资产名字

T_in, N = R_in.shape
print("In-sample shape:", R_in.shape)
print("Out-of-sample shape:", R_out.shape)
print("Assets:", list(tickers))


# ============================
# 2. 计算 in-sample 均值 & 协方差
# ============================
mu_in = R_in.mean(axis=0)  
mu_annual = (1 + mu_in) ** 12 - 1# 5 维均值向量
# Sigma_in = np.cov(R_in, rowvar=False) # 5x5 协方差矩阵
#print(Sigma_in)


#如果需要EWMA
lam = 0.97
T, N = R_in.shape

Sigma_in = np.zeros((N, N))
w = 1.0
weight_sum = 0.0

for t in range(T-1, -1, -1):
    r = R_in[t].reshape(-1, 1)
    Sigma_in += w * (r @ r.T)
    weight_sum += w
    w *= lam

Sigma_in = Sigma_in / weight_sum

Sigma_in_annual = Sigma_in * 12
print("Sigma_in:")
print(Sigma_in)
print("Sigma_in_annual:")
print(Sigma_in_annual)


w = np.array([0.30, 0.25, 0.20, 0.15, 0.10])  # 这是个例子，总和=1

rf = 0.04       # 年化无风险利率 4%
#rf = rf_annual / 12.0   # 月度无风险利率
#rf_daily = rf_annual / 365  #daily

#如果需要daily下面的一些parameter需要改名称，比如rf_daily
def port_stats(w, mu, Sigma, rf=0.0):
    """给定权重，返回 (期望收益, 波动率, Sharpe)"""
    w = np.array(w)
    ret = w @ mu
    vol = np.sqrt(w @ Sigma @ w)
    sharpe = (ret - rf) / vol
    return ret, vol, sharpe

ret_d, vol_d, sharpe_d = port_stats(w, mu_in, Sigma_in, rf)

print("===== Monthly Portfolio Stats =====")
print(f"Monthly Expected Return: {ret_d:.6f}")
print(f"Monthly Volatility:      {vol_d:.6f}")
print(f"Monthly Sharpe Ratio:    {sharpe_d:.6f}")


# ============================
# 3. 最大 Sharpe 组合 (in-sample)
# ============================
def neg_sharpe(w, mu, Sigma, rf):
    return -port_stats(w, mu, Sigma, rf)[2]

rf = 0.04
w0 = np.ones(N) / N
bounds = [(0, 1)] * N
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

res_msr = minimize(
    neg_sharpe,
    w0,
    args=(mu_in, Sigma_in, rf),
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)

w_msr = res_msr.x
ret_in_msr, vol_in_msr, sharpe_in_msr = port_stats(w_msr, mu_annual, Sigma_in_annual, rf)

print("\n===== (a) In-sample Max Sharpe Portfolio =====")
for name, weight in zip(tickers, w_msr):
    print(f"{name}: {weight:.4f}")
print(f"Expected monthly return: {ret_in_msr:.4%}")
print(f"Volatility (monthly):    {vol_in_msr:.4%}")
print(f"Sharpe (monthly):        {sharpe_in_msr:.4f}")

# ============================
# 3B. Risk Parity Portfolio
# ============================
def CSD(w, Sigma):
    """Component Std Dev (风险贡献的原始形式)"""
    w = np.array(w)
    denom = w @ Sigma @ w
    return (w * (Sigma @ w)) / denom

def rp_objective(w, Sigma):
    csd = CSD(w, Sigma)
    avg = np.mean(csd)
    return np.sum((csd - avg)**2)

w0 = np.ones(N) / N

res_rp = minimize(
    rp_objective,
    w0,
    args=(Sigma_in,),
    method="SLSQP",
    bounds=bounds,
    constraints=constraints
)

w_rp = res_rp.x
ret_in_rp, vol_in_rp, sharpe_in_rp = port_stats(w_rp, mu_annual, Sigma_in_annual, rf)

print("\n===== (c) In-sample Risk Parity Portfolio =====")
for name, weight in zip(tickers, w_rp):
    print(f"{name}: {weight:.4f}")
print(f"Expected monthly return: {ret_in_rp:.4%}")
print(f"Volatility (monthly):    {vol_in_rp:.4%}")
print(f"Sharpe (monthly):        {sharpe_in_rp:.4f}")

# ============================
# 3C. Risk Parity with Risk Budget
# ============================
risk_budget = np.array([1, 1, 1, 1, 0.5])  # ✅ 第 5 个资产风险贡献减半

def rp_budget_objective(w, Sigma, risk_budget):
    csd = CSD(w, Sigma)
    csd_adj = csd / risk_budget
    avg = np.mean(csd_adj)
    return np.sum((csd_adj - avg)**2)

res_rpb = minimize(
    rp_budget_objective,
    w0,
    args=(Sigma_in, risk_budget),
    method="SLSQP",
    bounds=bounds,
    constraints=constraints
)

w_rpb = res_rpb.x
ret_in_rpb, vol_in_rpb, sharpe_in_rpb = port_stats(w_rpb, mu_in, Sigma_in, rf)

print("\n===== (d) In-sample Risk Parity (Risk Budgeted) =====")
for name, weight in zip(tickers, w_rpb):
    print(f"{name}: {weight:.4f}")
print(f"Expected monthly return: {ret_in_rpb:.4%}")
print(f"Volatility (monthly):    {vol_in_rpb:.4%}")
print(f"Sharpe (monthly):        {sharpe_in_rpb:.4f}")

# ============================
# 4. Equal-weight 组合 (in-sample)
# ============================
w_eq = np.ones(N) / N
ret_in_eq, vol_in_eq, sharpe_in_eq = port_stats(w_eq, mu_in, Sigma_in, rf)

print("\n===== In-sample Equal-weight Portfolio =====")
print("Each asset weight: 0.2000")
print(f"Expected monthly return: {ret_in_eq:.4%}")
print(f"Volatility (monthly):    {vol_in_eq:.4%}")
print(f"Sharpe (monthly):        {sharpe_in_eq:.4f}")

# ============================
# 5. Out-of-sample 评价函数
# ============================
def oos_performance(w, R, rf):
    """给定权重 w 和 OOS 收益矩阵 R，返回 OOS 指标"""
    w = np.array(w)
    port_ret = R @ w          # 每期组合收益
    mean_r = port_ret.mean()
    vol = port_ret.std(ddof=1)
    sharpe = (mean_r - rf) / vol
    total = np.prod(1 + port_ret) - 1
    return mean_r, vol, sharpe, total

# ============================
# 6. (b) OOS 表现：Max Sharpe vs Equal-weight
# ============================
mean_msr, vol_msr, sharpe_msr, total_msr = oos_performance(w_msr, R_out, rf)
mean_eq,  vol_eq,  sharpe_eq,  total_eq  = oos_performance(w_eq,  R_out, rf)

print("\n===== (b) Out-of-sample Performance =====")
print("---- Max Sharpe portfolio ----")
print(f"Average monthly return: {mean_msr:.4%}")
print(f"Volatility (monthly):   {vol_msr:.4%}")
print(f"Sharpe (monthly):       {sharpe_msr:.4f}")
print(f"Total return (OOS):     {total_msr:.2%}")

print("\n---- Equal-weight portfolio ----")
print(f"Average monthly return: {mean_eq:.4%}")
print(f"Volatility (monthly):   {vol_eq:.4%}")
print(f"Sharpe (monthly):       {sharpe_eq:.4f}")
print(f"Total return (OOS):     {total_eq:.2%}")
