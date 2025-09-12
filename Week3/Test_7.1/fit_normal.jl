using Statistics, Distributions

function fit_normal(data::Vector)
    μ = mean(data)
    σ = std(data)
    return NamedTuple{(:errorModel,)}((Normal(μ, σ),))
end
