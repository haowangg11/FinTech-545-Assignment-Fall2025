using Statistics, Distributions

mutable struct FittedNormal
    μ::Float64
    σ::Float64
    errorModel::Normal
end

function fit_normal(x::Vector{Float64})
    μ = mean(x)
    σ = std(x)
    return FittedNormal(μ, σ, Normal(μ, σ))
end

function VaR(model::Normal; α::Float64 = 0.05)
    return -quantile(model, α)
end
