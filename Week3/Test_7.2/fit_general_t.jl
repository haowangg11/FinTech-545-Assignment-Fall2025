using Distributions, Optim

function fit_general_t(data::Vector)
    init = [mean(data), std(data), 10.0]

    function neg_loglik(params)
        μ, σ, ν = params
        if σ <= 0 || ν <= 2  
            return Inf
        end
        
        z = (data .- μ) ./ σ
        return -sum(logpdf.(TDist(ν), z)) + length(data)*log(σ)
    end

    
    result = optimize(neg_loglik, init, BFGS())
    p = Optim.minimizer(result)
    μ, σ, ν = p

    return (
        errorModel = LocationScale(μ, σ, TDist(ν)),
        ρ = TDist(ν)
    )
end
