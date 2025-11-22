import numpy as np
import pandas as pd
from numpy.linalg import lstsq

def expost_factor_11_2(start_weights, stock_returns, factor_returns, beta):
    # Convert to numpy
    stock_returns = np.asarray(stock_returns, dtype=float)
    factor_returns = np.asarray(factor_returns, dtype=float)
    beta = np.asarray(beta, dtype=float)
    w = np.asarray(start_weights, dtype=float)

    T, N = stock_returns.shape
    M = factor_returns.shape[1]

    # ===== 1. Portfolio returns & evolving weights ===== #
    weights = np.zeros((T, N))
    port_returns = np.zeros(T)
    weights[0] = w.copy()

    for t in range(T):
        Rt = float(np.dot(w, stock_returns[t]))
        port_returns[t] = Rt

        # next period weights
        w_star = w * (1.0 + stock_returns[t])
        if t < T - 1:
            w = w_star / (1.0 + Rt)
            weights[t + 1] = w

    # ===== 2. Factor weights & contrib ===== #
    factorWeights = weights @ beta                    # (T, M)
    contrib = factorWeights * factor_returns          # (T, M)
    alpha_t = port_returns - contrib.sum(axis=1)      # (T,)

    # ✅ ===== 3. TotalReturn (GEOMETRIC — correct per NOTE) ===== #
    # factor total returns
    TR_factor = np.exp(np.log(1.0 + factor_returns).sum(axis=0)) - 1.0
    # portfolio total return
    TR_portfolio = np.exp(np.log(1.0 + port_returns).sum()) - 1.0
    # alpha
    TR_alpha = np.exp(np.log(1.0 + alpha_t).sum()) - 1.0


    # ✅ ===== 4. Carino Linking (Return Attribution) ===== #
    GR = np.log(1.0 + TR_portfolio)
    K = GR / TR_portfolio
    k = np.log(1.0 + port_returns) / (K * port_returns)

    RA_factor = (contrib * k[:, None]).sum(axis=0)
    RA_alpha = float((alpha_t * k).sum())

    # ✅ ===== 5. Vol Attribution ===== #
    sigma_p = float(np.std(port_returns, ddof=1))

    Vol_factor = np.zeros(M)
    for j in range(M):
        beta_j = lstsq(port_returns.reshape(-1,1), contrib[:, j], rcond=None)[0][0]
        Vol_factor[j] = sigma_p * beta_j

    beta_alpha = lstsq(port_returns.reshape(-1,1), alpha_t, rcond=None)[0][0]
    Vol_alpha = sigma_p * beta_alpha
    Vol_portfolio = sigma_p

    # ===== 6. Final Output ===== #
    df = pd.DataFrame({
        "Value": ["TotalReturn", "Return Attribution", "Vol Attribution"],
        "F1": [TR_factor[0], RA_factor[0], Vol_factor[0]],
        "F2": [TR_factor[1], RA_factor[1], Vol_factor[1]],
        "F3": [TR_factor[2], RA_factor[2], Vol_factor[2]],
        "Alpha": [TR_alpha, RA_alpha, Vol_alpha],
        "Portfolio": [TR_portfolio, TR_portfolio, Vol_portfolio]
    })

    return df, weights, factorWeights


# ===== RUN ===== #
if __name__ == "__main__":
    stWgt = pd.read_csv("data/test11_2_weights.csv")["W"].values
    factor_returns = pd.read_csv("data/test11_2_factor_returns.csv").values
    stock_returns = pd.read_csv("data/test11_2_stock_returns.csv").values
    beta = pd.read_csv("data/test11_2_beta.csv").iloc[:,1:].values

    Attribution, weights, factorWeights = expost_factor_11_2(
        stWgt, stock_returns, factor_returns, beta
    )

    Attribution.to_csv("data/testout11_2.csv", index=False)
    print("✅ Saved to data/testout11_2.csv")
