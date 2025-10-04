using LinearAlgebra, Random

function simulateNormal(n::Int, cov::Matrix{Float64})
    d = size(cov, 1)

    cov = (cov + cov') / 2

    A = nothing 
    try
        A = cholesky(cov).L
    catch e
        @warn "Cholesky failed, applying nearPD correction..."
        λ, V = eigen(cov)
        λ = map(x -> max(x, 1e-10), λ)
        cov_fixed = V * Diagonal(λ) * V'
        A = cholesky(cov_fixed).L  
    end

    Z = randn(n, d)
    return Z * A'
end
