using LinearAlgebra, Random

function simulateNormal(n::Int, cov::Matrix{Float64})
    d = size(cov, 1)
    cov = (cov + cov') / 2

    λ, V = eigen(Symmetric(cov))  
    λ_fixed = map(x -> max(x, 1e-10), λ)
    cov_fixed = real.(V * Diagonal(λ_fixed) * V')

    cov_fixed = (cov_fixed + cov_fixed') / 2

    A = cholesky(Symmetric(cov_fixed)).L

    Z = randn(n, d)
    return Z * A'
end
