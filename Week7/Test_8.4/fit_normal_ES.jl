using CSV, DataFrames, Distributions, Statistics

function ES(dist::Distribution; α=0.05)
    VaR = quantile(dist, α)
    if dist isa Normal
        μ, σ = dist.μ, dist.σ
        pdf_ratio = pdf(Normal(0,1), quantile(Normal(0,1), α)) / α
        return -(μ - σ * pdf_ratio)
    else
        xs = quantile(dist, range(0, α, length=1000))
        return -mean(xs)
    end
end


function fit_normal(data::Vector)
    μ = mean(data)
    σ = std(data)
    return (errorModel = Normal(μ, σ),)
end
