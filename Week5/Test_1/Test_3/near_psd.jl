using LinearAlgebra

"""
near_psd(A; correlation=false)

修复矩阵 A，使其成为最近的正半定矩阵 (PSD)。
- correlation = true → 强制对角线为 1（用于相关矩阵）
- correlation = false → 保持原方差比例（用于协方差矩阵）
"""
function near_psd(A::Matrix{Float64}; correlation::Bool=false)
    # 对称化
    A = (A + A') / 2
    vals, vecs = eigen(A)
    vals[vals .< 1e-10] .= 1e-10       # 抬升负特征值
    A_psd = vecs * Diagonal(vals) * vecs'

    if correlation
        # correlation matrix: 对角线=1
        D = Diagonal(1 ./ sqrt.(diag(A_psd)))
        A_psd = D * A_psd * D
    else
        # covariance matrix: 保持原始方差比例
        D = Diagonal(sqrt.(diag(A)) ./ sqrt.(diag(A_psd)))
        A_psd = D * A_psd * D
    end

    return (A_psd + A_psd') / 2
end
