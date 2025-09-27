using LinearAlgebra, Statistics
@info "✅ ewCovar.jl (vectorized, normalized, weighted-mean)"

"""
    ewCovar(x, λ)

向量化的 EWMA 协方差计算（一次矩阵运算搞定）：
S = X_c' * W * X_c
其中：
- W 为对角权重矩阵，w_t = (1-λ) * λ^(n-t)，并做归一化使 ∑w_t = 1
- μ_w 为同一组权重下的**加权均值**，X_c = X - 1*μ_w

这样既符合 RiskMetrics 思想（指数衰减），又避免尺度爆炸。
"""
function ewCovar(x, λ::Float64)
    # 1) 强转 Float64，兼容 Matrix{Any} / DataFrame -> Matrix(x)
    X = Matrix{Float64}(x)
    n, d = size(X)

    # 2) 生成归一化的指数权重（最新样本权重最大）
    #    t = 1..n 对应权重 w_t ∝ λ^(n - t)
    w = λ .^ (n .- collect(1:n))
    w .*= (1 - λ)             # 标准 EWMA 的 (1-λ) 系数
    w ./= sum(w)              # 归一化，确保 ∑w = 1

    # 3) 加权均值（按列）：μ_w = ∑_t w_t * x_t
    #    用矩阵乘：μ_w(1×d) = w(1×n) * X(n×d)
    μw = (w') * X                         # 1×d
    # 4) 去均值：X_c = X - 1*μ_w
    Xc = X .- ones(n) * μw                # n×d

    # 5) S = X_c' * W * X_c
    #    先把 w 当作逐行缩放：W*X_c 等价于 eachrow 乘以 w_t
    WXc = Xc .* w                         # n×d（逐行乘权重）
    S = Xc' * WXc                         # d×d

    return S
end
