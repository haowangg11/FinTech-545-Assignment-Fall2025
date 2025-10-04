using LinearAlgebra, Random

function simulate_pca(Σ::Matrix{Float64}, n::Int; pctExp::Float64 = 0.99)
    Random.seed!(5454)
    λ, V = eigen(Symmetric((Σ + Σ') / 2))
    idx = sortperm(λ, rev = true)
    λ = λ[idx]
    V = V[:, idx]

    λ_norm = λ / sum(λ)
    cumexp = cumsum(λ_norm)
    k = findfirst(x -> x >= pctExp, cumexp)
    k = isnothing(k) ? length(λ) : k

    L = V[:, 1:k] * Diagonal(sqrt.(λ[1:k]))

    Z = randn(n, k)
    X = Z * L'

    return X
end
