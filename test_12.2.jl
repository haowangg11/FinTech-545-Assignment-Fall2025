using CSV, DataFrames, FiniteDiff

function bt_american(call::Bool, underlying, strike, ttm, rf, b, ivol, N)
    dt = ttm / N
    u = exp(ivol * sqrt(dt))
    d = 1 / u
    pu = (exp(b * dt) - d) / (u - d)
    pd = 1.0 - pu
    df = exp(-rf * dt)
    z = call ? 1 : -1

    # helper index functions
    nNodeFunc(n) = convert(Int64, (n + 1) * (n + 2) / 2)
    idxFunc(i, j) = nNodeFunc(j - 1) + i + 1
    nNodes = nNodeFunc(N)

    optionValues = Vector{Float64}(undef, nNodes)

    # backward induction
    for j in N:-1:0
        for i in j:-1:0
            idx = idxFunc(i, j)
            price = underlying * u^i * d^(j - i)
            # payoff
            optionValues[idx] = max(0, z * (price - strike))

            if j < N
                continuation = df * (pu * optionValues[idxFunc(i + 1, j + 1)] +
                                     pd * optionValues[idxFunc(i, j + 1)])
                optionValues[idx] = max(optionValues[idx], continuation)
            end
        end
    end

    return optionValues[1]
end


options = filter(r -> !ismissing(r.ID),
    CSV.read("data/test12_1.csv", DataFrame))

# compute values
outVals = [bt_american(o["Option Type"] == "Call",
                       o.Underlying,
                       o.Strike,
                       o.DaysToMaturity / o.DayPerYear,
                       o.RiskFreeRate,
                       o.RiskFreeRate - o.DividendRate,  # b = r - q
                       o.ImpliedVol,
                       500) for o in eachrow(options)]

# wrapper functions for gradient calculations
function fcall(_p)
    p = collect(_p)
    bt_american(true, p[1], p[2], p[3], p[4], p[5], p[6], 500)
end

function fput(_p)
    p = collect(_p)
    bt_american(false, p[1], p[2], p[3], p[4], p[5], p[6], 500)
end

# initialize greek arrays
deltas = Float64[]
gammas = Float64[]
vegas = Float64[]
rhos = Float64[]
thetas = Float64[]

# loop each option
for o in eachrow(options)
    parms = [o.Underlying,
             o.Strike,
             o.DaysToMaturity / o.DayPerYear,
             o.RiskFreeRate,
             o.RiskFreeRate - o.DividendRate,
             o.ImpliedVol]

    if o["Option Type"] == "Call"
        v = fcall(parms)
        grad = FiniteDiff.finite_difference_gradient(fcall, parms)

        push!(deltas, grad[1])
        d = 1.5
        parms[1] += d
        gamma1 = fcall(parms)
        parms[1] -= 2d
        gamma2 = fcall(parms)
        gamma = (gamma1 + gamma2 - 2 * v) / (d^2)
        push!(gammas, gamma)

        push!(vegas, grad[6])
        push!(rhos, grad[4])
        push!(thetas, grad[3])

    else  # Put
        v = fput(parms)
        grad = FiniteDiff.finite_difference_gradient(fput, parms)

        push!(deltas, grad[1])
        d = 1.5
        parms[1] += d
        gamma1 = fput(parms)
        parms[1] -= 2d
        gamma2 = fput(parms)
        gamma = (gamma1 + gamma2 - 2 * v) / (d^2)
        push!(gammas, gamma)

        push!(vegas, grad[6])
        push!(rhos, grad[4])
        push!(thetas, grad[3])
    end
end

# write output file
CSV.write("data/testout12_2.csv",
    DataFrame(ID = options.ID,
              Value = outVals,
              Delta = deltas,
              Gamma = gammas,
              Vega = vegas,
              Rho = rhos,
              Theta = thetas))

