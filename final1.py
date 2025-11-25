import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib.pyplot as plt

def bs_call(S, K, T, r, q, sigma):
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def bs_put(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)


def implied_vol_call(S, K, T, r, q, market_price):
    func = lambda sigma: bs_call(S, K, T, r, q, sigma) - market_price
    return brentq(func, 1e-6, 5.0)

#iv = implied_vol_call(S, K, T, r, q, call_price)
#print("Implied Volatility:", iv)

df = pd.read_csv("data/problem1.csv")
returns = df.values.flatten()
trading_days = 255
sigma_1 = np.std(returns, ddof=1)
sigma_annual = sigma_1 * np.sqrt(trading_days)
print("vol:", sigma_1)
print("Annual vol:", sigma_annual)
print("put at strike is 99:")
p_99 = bs_put(100, 99, 1/255, 0.04, sigma_1)
print(p_99)
print("put at strike is 100:")
p_100 = bs_put(100, 100, 1/255, 0.04, sigma_1)
print(p_100)
print("put at strike is 101:")
p_101 = bs_put(100, 101, 1/255, 0.04, sigma_1)
print(p_101)
print("call at strike is 99:")
c_99 = bs_call(100, 99, 1/255, 0.04, 0, sigma_1)
print(c_99)
print("put at strike is 100:")
c_100 = bs_call(100, 100, 1/255, 0.04, 0, sigma_1)
print(c_100)
print("put at strike is 101:")
c_101 = bs_call(100, 101, 1/255, 0.04, 0, sigma_1)
print(c_101)

Ks = np.arange(95, 106)
ivs = []
for K in Ks:
    market_price = bs_call(100, K, 1/255, 0.04, 0, sigma) 
    ivs.append(implied_vol_call(100, K, 1/255, 0.04, 0, market_price))

plt.plot(Ks, ivs, marker='o')
plt.xlabel("k")
plt.ylabel("iv")
plt.grid(True)
plt.show()
