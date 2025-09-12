using Statistics, Distributions

function fit_normal(data::Vector)
    μ = mean(data)
    σ = std(data)
    return Normal(μ, σ)
end
