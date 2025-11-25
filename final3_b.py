import numpy as np
import pandas as pd

#这个代码解决了2024fall exam的 5a&5b

def expost_11_1(start_weights, returns):
    returns = np.array(returns, dtype=float)
    w = np.array(start_weights, dtype=float)

    T, N = returns.shape
    weights = np.zeros((T, N))
    port_returns = np.zeros(T)

    weights[0] = w.copy()

    # ---- Step 1-3: portfolio return & evolving weights ---- #
    for t in range(T):
        Rt = np.dot(w, returns[t])
        port_returns[t] = Rt

        w_star = w * (1 + returns[t])
        if t < T - 1:
            w = w_star / (1 + Rt)
            weights[t+1] = w

    # ---- Total Return ---- #
    TotalReturn_asset = np.prod(1 + returns, axis=0) - 1
    TotalReturn_port  = np.prod(1 + port_returns) - 1

    # ---- Carino K ---- #
    GR = np.log(1 + TotalReturn_port)
    K = GR / TotalReturn_port
    k = np.log(1 + port_returns) / (K * port_returns)

    # ---- Return Attribution ---- #
    RA = (weights * returns) * k[:, None]
    RA_sum = RA.sum(axis=0)
    RA_port = TotalReturn_port

    # ✅ ---- Vol Attribution (CORRECT NOTE FORMULA) ---- #
    sigma_p = np.std(port_returns, ddof=1)

    VolAttr = np.zeros(N)
    for i in range(N):
        wr = weights[:, i] * returns[:, i]
        beta_i = np.cov(wr, port_returns)[0, 1] / np.var(port_returns, ddof=1)
        VolAttr[i] = sigma_p * beta_i

    VolAttr_port = sigma_p

    # ---- Output in correct column order ---- #
    df = pd.DataFrame({
        "Value": ["TotalReturn", "Return Attribution", "Vol Attribution"]
    })

    for i in range(N):
        df[f"x{i+1}"] = [
            TotalReturn_asset[i],
            RA_sum[i],
            VolAttr[i]
        ]

    df["Portfolio"] = [
        TotalReturn_port,
        RA_port,
        VolAttr_port
    ]

    return df


stWgt = np.array([0, 0, 0, 1, 0])
#equal weight stWgt = np.ones(5) / 5

df = pd.read_csv("data/problem3_outsample.csv")
returns = df.drop(columns=["Date"]).values 


output = expost_11_1(stWgt, returns)
output.to_csv("data/final_question3_b.csv", index=False)


print("✅ Saved final_question3_b.csv")
# 这个代码的输出归因那两行的前三列加起来等于portfolio对应的数值