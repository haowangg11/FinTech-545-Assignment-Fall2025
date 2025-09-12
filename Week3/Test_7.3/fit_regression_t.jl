using Optim, Distributions, LinearAlgebra

"""
    fit_regression_t(y, X)

Fits a regression model with T-distributed errors:
    y = α + X * β + ε
    ε ~ T_ν(μ, σ)

Returns:
    - beta: [α, β1, β2, β3]
    - errorModel.μ
    - errorModel.σ
    - ρ.ν (degrees of freedom)
"""
function fit_regression_t(y::Vector, X::Matrix)
    n, p = size(X)
    X_ = hcat(ones(n), X) 
    
    init_beta = X_ \ y
    init = vcat(init_beta, 0.0, std(y), 5.0)

    function neg_loglik(params)
        β = params[1:p+1]
        μ = params[p+2]
        σ = params[p+3]
        ν = params[p+4]

        if σ <= 0 || ν <= 2
            return Inf 
        end

        ε = y .- X_ * β
        z = (ε .- μ) ./ σ
        return -sum(logpdf.(TDist(ν), z)) + n * log(σ)
    end

    result = optimize(neg_loglik, init, BFGS())
    opt_params = Optim.minimizer(result)

    β = opt_params[1:p+1]
    μ = opt_params[p+2]
    σ = opt_params[p+3]
    ν = opt_params[p+4]

    return (
        beta = β,
        errorModel = LocationScale(μ, σ, TDist(ν)),
        ρ = TDist(ν)
    )
end
