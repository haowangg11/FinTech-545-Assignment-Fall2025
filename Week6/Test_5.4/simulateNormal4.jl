using LinearAlgebra, Random

function higham_nearestPSD(S::Matrix{Float64};
                           eps::Float64 = 0.0,   
                           max_iter::Int = 500,
                           tol::Float64 = 1e-12)
    A  = (S + S') / 2
    sd = sqrt.(diag(A));  sd = map(x -> max(x, 1e-12), sd)
    D½  = Diagonal(sd)
    D⁻½ = Diagonal(1.0 ./ sd)

    R = D⁻½ * A * D⁻½
    R = (R + R') / 2

    X = copy(R)
    Δ = zeros(size(R))
    for _ in 1:max_iter
        Y = X - Δ
        λ, V = eigen(Symmetric(Y))
        λ = map(x -> max(x, eps), λ)       
        P = V * Diagonal(λ) * V'
        P = (P + P') / 2
        Δ = P - Y

        @inbounds for i in 1:size(P,1)     
            P[i,i] = 1.0
        end

        if norm(P - X, Inf) < tol
            X = P
            break
        end
        X = P
    end
    @inbounds for i in 1:size(X,1)      
        X[i,i] = 1.0
    end

    Sfix = D½ * X * D½
    return (Sfix + Sfix') / 2
end

function simulateNormal(n::Int, cov::Matrix{Float64}; fixMethod = nothing)

    Random.seed!(5454)

    Σ = (cov + cov') / 2
    if fixMethod === higham_nearestPSD
        Σ = higham_nearestPSD(Σ)
    end

    Σ = (Σ + Σ') / 2 + 1e-12I

    L = cholesky(Symmetric(Σ)).L
    Z = randn(n, size(Σ,1))      
    return Z * L'
end
