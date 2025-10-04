using Statistics, Distributions, Optim

struct FittedT
    ρ::NamedTuple{(:ν,),Tuple{Float64}}
    σ::Float64
    μ::Float64
end

struct FitResultT
    errorModel::FittedT
end


function fit_general_t(x::Vector{Float64})
    μ = mean(x)
    σ = std(x)


    function negloglik(v)
        ν = v[1]
        if ν <= 2
            return Inf 
        end
        return -sum(logpdf.(TDist(ν), (x .- μ) ./ σ))
    end

    result = optimize(negloglik, [5.0])  
    ν̂ = Optim.minimizer(result)[1]

    println("Estimated ν = ", ν̂)

    return FitResultT(FittedT((ν=ν̂,), σ, μ))
end


function VaR(model::FittedT; α::Float64=0.05)
    q = quantile(TDist(model.ρ.ν), α)
    return -(model.μ + model.σ * q)
end
