using CSV, DataFrames, Distributions

function gbsm(is_call::Bool, S, K, T, r, q, σ; includeGreeks=false)
    N = Normal()
    d1 = (log(S / K) + (r - q + 0.5 * σ^2) * T) / (σ * sqrt(T))
    d2 = d1 - σ * sqrt(T)
    if is_call
        value = S * exp(-q * T) * cdf(N, d1) - K * exp(-r * T) * cdf(N, d2)
        delta = exp(-q * T) * cdf(N, d1)
        theta = -(S * σ * exp(-q * T) * pdf(N, d1)) / (2 * sqrt(T)) - r * K * exp(-r * T) * cdf(N, d2) + q * S * exp(-q * T) * cdf(N, d1)
    else
        value = K * exp(-r * T) * cdf(N, -d2) - S * exp(-q * T) * cdf(N, -d1)
        delta = exp(-q * T) * (cdf(N, d1) - 1)
        theta = -(S * σ * exp(-q * T) * pdf(N, d1)) / (2 * sqrt(T)) + r * K * exp(-r * T) * cdf(N, -d2) - q * S * exp(-q * T) * cdf(N, -d1)
    end
    gamma = exp(-q * T) * pdf(N, d1) / (S * σ * sqrt(T))
    vega = S * exp(-q * T) * pdf(N, d1) * sqrt(T)
    rho = (is_call ? K * T * exp(-r * T) * cdf(N, d2) : -K * T * exp(-r * T) * cdf(N, -d2))
    if includeGreeks
        return (value=value, delta=delta, gamma=gamma, vega=vega, rho=rho, theta=theta)
    else
        return value
    end
end

options = filter(r -> !ismissing(r.ID), CSV.read("data/test12_1.csv", DataFrame))
outVals = [gbsm(o["Option Type"] == "Call", o.Underlying, o.Strike, o.DaysToMaturity / o.DayPerYear, o.RiskFreeRate, o.DividendRate, o.ImpliedVol; includeGreeks=true) for o in eachrow(options)]
values = [v.value for v in outVals]
deltas = [v.delta for v in outVals]
gammas = [v.gamma for v in outVals]
vegas = [v.vega for v in outVals]
rhos = [v.rho for v in outVals]
thetas = [v.theta for v in outVals]
CSV.write("data/testout12_1.csv", DataFrame(:ID => options.ID, :Value => values, :Delta => deltas, :Gamma => gammas, :Vega => vegas, :Rho => rhos, :Theta => thetas))
