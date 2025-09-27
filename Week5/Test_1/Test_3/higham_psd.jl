using LinearAlgebra

"""
higham_nearestPSD(A; correlation=false, tol=1e-8, max_iter=100)

Higham (2002) 最近正半定矩阵算法。
- correlation=true → 对角线保持为1（相关矩阵）
"""
function higham_nearestPSD(A::Matrix{Float64}; correlation::Bool=false, tol=1e-8, max_iter=100)
    n = size(A, 1)
    Y = copy(A)
    ΔS = zeros(n, n)

    for _ in 1:max_iter
        R = Y - ΔS
        vals, vecs = eigen((R + R') / 2)
        vals[vals .< 1e-10] .= 1e-10
        X = vecs * Diagonal(vals) * vecs'
        ΔS = X - R
        Y = X
        if norm(ΔS, Inf) < tol
            break
        end
    end

    # 对称化 + 相关矩阵时对角线设为 1
    Y = (Y + Y') / 2
    if correlation
        D = Diagonal(1 ./ sqrt.(diag(Y)))
        Y = D * Y * D
    else
        D = Diagonal(sqrt.(diag(A)) ./ sqrt.(diag(Y)))
        Y = D * Y * D
    end

    return (Y + Y') / 2
end
