using LinearAlgebra

function higham_nearest_psd(A::Matrix{Float64}; tol=1e-8, max_iter=100)
    n = size(A, 1)
    X = copy(A)
    Y = zeros(n, n)
    for _ in 1:max_iter
        R = X - Y
        eigvals, eigvecs = eigen(Symmetric(R))
        eigvals[eigvals .< 0.0] .= 0.0
        Xp = eigvecs * Diagonal(eigvals) * eigvecs'
        Y = Xp - R
        X = copy(Xp)
        if minimum(eigvals) > -tol
            break
        end
    end
    return (X + X') / 2
end


function simulateNormal(n::Int64, cov::Matrix{Float64}; fixMethod=:none)
    if fixMethod == :near_psd
        cov = higham_nearest_psd(cov)
    end

    cov = (cov + cov') / 2

    λmin = minimum(eigvals(cov))
    println("λmin = ", λmin)

    if λmin <= 0
        δ = abs(λmin) + 1e-6
        println("δ = ", δ)
        cov += δ * I
    end

    L = nothing
    try
        L = cholesky(Symmetric(cov)).L
    catch
        println("(1e-3 * I)")
        cov += 1e-3 * I
        try
            L = cholesky(Symmetric(cov)).L
        catch
            println("final repair")
            cov = Diagonal(diag(cov) .+ 1e-3)
            L = cholesky(Symmetric(cov)).L
        end
    end

    Z = randn(n, size(cov, 1))
    X = Z * L'
    return X
end
