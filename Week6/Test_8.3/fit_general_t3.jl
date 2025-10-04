using Distributions, Optim, Statistics

mutable struct FittedT
    μ::Float64
    σ::Float64
    ρ::NamedTuple
    eval::Function   # ✅ 新增 eval 函数成员
end


function fit_general_t(x::Vector{Float64})
    μ0 = mean(x)
    σ0 = std(x)
    ν0 = 5.0

    function negloglik(params)
        μ, σ, ν = params
        σ <= 0 && return Inf
        ν <= 2 && return Inf
        sum(-logpdf(TDist(ν), (x .- μ) ./ σ) .+ log(σ))
    end

    lower_bounds = [-Inf, 1e-6, 2.01]
    upper_bounds = [Inf, Inf, 100.0]
    initial = [μ0, σ0, ν0]

    result = optimize(negloglik, lower_bounds, upper_bounds, initial,
                      Fminbox(LBFGS()); autodiff = :forward)
    μ̂, σ̂, ν̂ = Optim.minimizer(result)

    eval_fn = z -> μ̂ .+ σ̂ .* rand(TDist(ν̂), length(z))

    return FittedT(μ̂, σ̂, (ν = ν̂,), eval_fn)
end


function VaR(fd::FittedT; α::Float64 = 0.05)
    quantile(TDist(fd.ρ.ν), α) * fd.σ + fd.μ
end

function VaR(x::Vector{Float64}; α::Float64 = 0.05)
    return -quantile(x, α)
end