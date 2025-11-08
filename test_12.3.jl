using CSV, DataFrames

function bt_american_cont(call::Bool, S, K, T, r, b, σ, N)
    dt = T / N
    u  = exp(σ * sqrt(dt))
    d  = 1 / u
    pu = (exp(b * dt) - d) / (u - d)
    pd = 1.0 - pu
    df = exp(-r * dt)
    z  = call ? 1 : -1

    nNode(n) = Int((n + 1) * (n + 2) ÷ 2)
    idx(i, j) = nNode(j - 1) + i + 1

    values = Vector{Float64}(undef, nNode(N))
    for j in N:-1:0
        for i in j:-1:0
            k   = idx(i, j)
            Sij = S * u^i * d^(j - i)
            ex  = max(0, z * (Sij - K))
            if j == N
                values[k] = ex
            else
                cont = df * (pu * values[idx(i + 1, j + 1)] + pd * values[idx(i, j + 1)])
                values[k] = max(ex, cont)
            end
        end
    end
    return values[1]
end

function bt_american(call::Bool, S, K, T, r,
                     divAmts::Vector{Float64}, divTimes::Vector{Int64},
                     σ, N)

    if isempty(divAmts) || isempty(divTimes)
        return bt_american_cont(call, S, K, T, r, r, σ, N)
    end
    if divTimes[1] > N
        return bt_american_cont(call, S, K, T, r, r, σ, N)
    end

    dt = T / N
    u  = exp(σ * sqrt(dt))
    d  = 1 / u
    pu = (exp(r * dt) - d) / (u - d)
    pd = 1.0 - pu
    df = exp(-r * dt)
    z  = call ? 1 : -1

    nNode(n) = Int((n + 1) * (n + 2) ÷ 2)
    idx(i, j) = nNode(j - 1) + i + 1

    J  = divTimes[1]
    nD = length(divTimes)
    values = Vector{Float64}(undef, nNode(J))

    for j in J:-1:0
        for i in j:-1:0
            k   = idx(i, j)
            Sij = S * u^i * d^(j - i)
            ex  = max(0, z * (Sij - K))

            if j < J
                cont = df * (pu * values[idx(i + 1, j + 1)] + pd * values[idx(i, j + 1)])
                values[k] = max(ex, cont)
            else
                nextS   = Sij - divAmts[1]
                nextT   = T - J * dt
                nextN   = N - J
                nextA   = nD > 1 ? divAmts[2:end] : Float64[]
                nextTau = nD > 1 ? (divTimes[2:end] .- divTimes[1]) : Int64[]
                cont_after_div = bt_american(call, nextS, K, nextT, r,
                                             nextA, nextTau, σ, nextN)
                values[k] = max(ex, cont_after_div)
            end
        end
    end

    return values[1]
end

options = filter(r -> !ismissing(r.ID), CSV.read("data/test12_3.csv", DataFrame))
options.DividendDates = [parse.(Int, v) for v in split.(options.DividendDates, ",")]
options.DividendAmts  = [parse.(Float64, v) for v in split.(options.DividendAmts, ",")]
options.N             = options.DaysToMaturity .* 2
options.DividendDates = options.DividendDates .* 2

outVals = [bt_american(o["Option Type"] == "Call", o.Underlying, o.Strike,
                       o.DaysToMaturity/o.DayPerYear, o.RiskFreeRate,
                       o.DividendAmts, o.DividendDates, o.ImpliedVol, o.N)
           for o in eachrow(options)]

CSV.write("data/testout12_3.csv", DataFrame(ID=options.ID, Value=outVals))

