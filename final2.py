import pandas as pd
import numpy as np
from scipy.stats import norm, t
from scipy.optimize import minimize

df = pd.read_csv("data/problem2.csv")

prices = df["SPY"].values

arth_ret = prices[1:] / prices[0:-1] - 1
log_ret  = np.log(prices[1:] / prices[0:-1])

print(" Arithmetic Returns (first 5):")
print(arth_ret[:5])

print("\n Log Returns (first 5):")
print(log_ret[:5])

# Fit Normal Distribution
mu_norm = np.mean(arth_ret)
sigma_norm = np.std(arth_ret, ddof=1)

print("\n Normal Fit Parameters:")
print(f"μ = {mu_norm:.6f}")
print(f"σ = {sigma_norm:.6f}")

# Log-likelihood for normal
ll_norm = np.sum(norm(mu_norm, sigma_norm).logpdf(arth_ret))
print(f"Log-Likelihood (Normal) = {ll_norm:.2f}")


# Fit Student-t (MLE)
def t_neg_ll(params, x):
    mu, sigma, nu = params
    if sigma <= 0 or nu <= 2:
        return 1e10
    return -np.sum(t(df=nu, loc=mu, scale=sigma).logpdf(x))

# Initial guesses
init = [mu_norm, sigma_norm, 5.0]
bounds = [(None, None), (1e-6, None), (2.01, None)]

res = minimize(t_neg_ll, init, args=(arth_ret,), method="L-BFGS-B", bounds=bounds)
mu_t, sigma_t, nu_t = res.x

print("\n Student-t Fit Parameters:")
print(f"μ = {mu_t:.6f}")
print(f"σ = {sigma_t:.6f}")
print(f"ν = {nu_t:.6f}")

# Log-likelihood for t
ll_t = -t_neg_ll([mu_t, sigma_t, nu_t], arth_ret)
print(f"Log-Likelihood (Student-t) = {ll_t:.2f}")

print("\n Conclusion:")
if ll_t > ll_norm:
    print(" Student-t provides a better fit (heavier tails).")
else:
    print(" Normal provides a better fit.")
    
"""
# =========================
# ✅ 6. 95% VaR (Normal & t)
# =========================
alpha = 0.95

VaR_norm = -(mu_norm + sigma_norm * norm.ppf(1 - alpha))
VaR_t    = -(mu_t + sigma_t * t.ppf(1 - alpha, df=nu_t))

print("\n===== 95% VaR =====")
print(f"VaR (Normal)    = {VaR_norm:.5f}")
print(f"VaR (Student-t) = {VaR_t:.5f}")

# =========================
# ✅ 7. 95% ES (Normal & t)
# =========================
# Closed form ES for Normal
ES_norm = -(mu_norm - sigma_norm * norm.pdf(norm.ppf(1 - alpha)) / (1 - alpha))

# ES via Monte Carlo for Student-t
sim = t.rvs(df=nu_t, loc=mu_t, scale=sigma_t, size=100000)
VaR_sim = np.quantile(sim, 1 - alpha)
ES_t = -sim[sim <= VaR_sim].mean()

print("\n===== 95% ES =====")
print(f"ES (Normal)     = {ES_norm:.5f}")
print(f"ES (Student-t)  = {ES_t:.5f}")
"""

#以下是如果没有sigma
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

def bs_call(S, K, T, r, q, sigma):
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def implied_vol_call(S, K, T, r, q, market_price):
    func = lambda sigma: bs_call(S, K, T, r, q, sigma) - market_price
    return brentq(func, 1e-6, 5.0)

# ---- Input from CSV ----
S = 665
K = 665
T = 10/255
r = 0.04
q = 0.0109
call_price = 7.05
put_price = 7.69

sigma_iv = implied_vol_call(S, K, T, r, q, call_price)
sigma_iv_put = implied_vol_call(S, K, T, r, q, put_price)

print("\n=== Implied volatility from Problem 2 ===")
print("sigma_IV =", sigma_iv)
print("signma_IV_put=", sigma_iv_put)

# ---------------------------
# 1️⃣ 定义 GBSM 函数
# ---------------------------
def gbsm(is_call, S, K, T, r, q, sigma):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if is_call:
        value = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = np.exp(-q * T) * norm.cdf(d1)
        theta = (-(S * sigma * np.exp(-q * T) * norm.pdf(d1)) / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2)
                 + q * S * np.exp(-q * T) * norm.cdf(d1))
    else:
        value = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        delta = np.exp(-q * T) * (norm.cdf(d1) - 1)
        theta = (-(S * sigma * np.exp(-q * T) * norm.pdf(d1)) / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)
                 - q * S * np.exp(-q * T) * norm.cdf(-d1))

    gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
    rho = (K * T * np.exp(-r * T) * norm.cdf(d2)
           if is_call else
           -K * T * np.exp(-r * T) * norm.cdf(-d2))

    return value, delta, gamma, vega, rho, theta

#第二题 c问
day_per_year = 255.0
TTM = 10
T0 = TTM / day_per_year

# 今天组合价值：long 1 call + 1 put
call0, _, _, _, _, _ = gbsm(True,  S, K, T0, r, q, sigma_iv)
put0,  _, _, _, _, _ = gbsm(False, S, K, T0, r, q, sigma_iv_put)

V0 = -call0 + put0 + S
print(f"\nPortfolio value today (1 call + 1 put): {V0:.4f}")

T1_days = TTM - 3
T1 = T1_days / day_per_year

call1, _, _, _, _, _ = gbsm(True,  S, K, T1, r, q, sigma_iv)
put1,  _, _, _, _, _ = gbsm(False, S, K, T1, r, q, sigma_iv_put)
V1 = -call1 + put1 + S

PL_100 = V1 - V0

print("\n=== Q3(a): 100-day buy-and-hold P/L (no other changes) ===")
print(f"Value after 100 days (same S, sigma, r, q): {V1:.4f}")
print(f"Profit / Loss over 100 days: {PL_100:.4f}")


# 也可以顺便给出 1-day 期望收益\
    
# ---- t 分布的 log-likelihood ----
def general_t_ll(params, x):
    mu, sigma, nu = params
    # 避免非法参数
    if sigma <= 0 or nu <= 2:
        return 1e10
    dist = t(df=nu, loc=mu, scale=sigma)
    ll = np.sum(dist.logpdf(x))
    return -ll   # minimize negative log-likelihood

def fit_general_t(x):
    x = np.array(x, dtype=float)

    # Initial guesses（和你之前写的一样思路）
    start_m   = np.mean(x)
    start_nu  = 6.0 / pd.Series(x).kurt() + 4
    start_s   = np.sqrt(np.var(x) * (start_nu - 2) / start_nu)
    init      = np.array([start_m, 1.0, start_nu])

    bounds = [
        (None,  None),   # mu
        (1e-6,  None),   # sigma
        (2.0001, None)   # nu
    ]

    res = minimize(
        general_t_ll,
        init,
        args=(x,),
        method='L-BFGS-B',
        bounds=bounds
    )

    mu_hat, sigma_hat, nu_hat = res.x

    dist = t(df=nu_hat, loc=mu_hat, scale=sigma_hat)
    eval_fn = lambda u: dist.ppf(u)

    return {
        "errorModel": {
            "rho":   {"nu": nu_hat},
            "mu":    mu_hat,
            "sigma": sigma_hat
        },
        "eval": eval_fn,
        "dist": dist
    }
    
fd_t = fit_general_t(arth_ret)
dist_t = fd_t["dist"]
print("=== Problem 1: fitted t parameters ===")
print(fd_t["errorModel"])


# VaR & ES 

N = 10000  
R_sim = dist_t.rvs(size=N)

S1 = S * np.exp(R_sim)

T1_1day = (TTM - 1) / day_per_year

PL_1day = np.zeros(N)

for i in range(N):
    c1, _, _, _, _, _ = gbsm(True,  S1[i], K, T1_1day, r, q, sigma_iv)
    p1, _, _, _, _, _ = gbsm(False, S1[i], K, T1_1day, r, q, sigma_iv)
    V1_path = -c1 + p1 + S1
    PL_1day[i] = V1_path - V0  # 按“+赚钱 -赔钱”的 convention

# VaR/ES 通常用“损失为正”的定义
alpha = 0.05
losses = -PL_1day  # loss > 0 表示亏钱

VaR_5 = np.quantile(losses, alpha)
tail_losses = losses[losses >= VaR_5]   # 左尾（大亏损）
ES_5 = tail_losses.mean()

print("\n=== Q3(b): 1-day 5% VaR and ES (using t distribution) ===")
print(f"5% VaR  (loss, + = bad): {VaR_5:.4f}")
print(f"5% ES   (loss, + = bad): {ES_5:.4f}")