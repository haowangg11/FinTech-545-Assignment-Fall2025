using Distributions, Optim

"""
    fit_general_t(data::Vector)

Fits a scaled + shifted T distribution to the data:
    X ~ μ + σ * T_ν
Returns .errorModel with μ and σ, and .ρ with ν
"""
function fit_general_t(data::Vector)
    # 初始值设置：μ取均值，σ取标准差，ν设为10
    init = [mean(data), std(data), 10.0]

    # 目标函数：负对数似然
    function neg_loglik(params)
        μ, σ, ν = params
        if σ <= 0 || ν <= 2   # σ必须正，ν>2才有有限方差
            return Inf
        end
        # 标准化数据
        z = (data .- μ) ./ σ
        return -sum(logpdf.(TDist(ν), z)) + length(data)*log(σ)
    end

    # 优化
    result = optimize(neg_loglik, init, BFGS())
    p = Optim.minimizer(result)
    μ, σ, ν = p

    return (
        errorModel = LocationScale(μ, σ, TDist(ν)),
        ρ = TDist(ν)
    )
end
