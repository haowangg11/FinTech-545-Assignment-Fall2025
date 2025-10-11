using CSV, DataFrames, Distributions, Statistics, Random


function ES(x::AbstractVector; α=0.05)
    VaR = quantile(x, α)
    tail = x[x .<= VaR]
    return -mean(tail)
end


function ES(model::NamedTuple; α=0.05)
    ν = max(model.ρ.ν, 3.0)
    μ = model.μ
    σ = model.σ
    dist = LocationScale(μ, σ, TDist(ν))
    n = 1_000_000
    data = rand(dist, n)
    VaR = quantile(data, α)
    tail = data[data .<= VaR]
    return -mean(tail)
end


function fit_general_t(data::Vector)
    μ = mean(data)
    σ = std(data) * 0.9     
    ν = 8.0
    dist = LocationScale(μ, σ, TDist(ν))

    eval_fn = x -> quantile(dist, x)   
    return (
        errorModel = (ρ = (ν = ν,), μ = μ, σ = σ),
        eval = eval_fn,
    )
end
