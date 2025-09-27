using CSV, DataFrames, LinearAlgebra
include("ewCovar.jl")

x = CSV.read("data/test2.csv", DataFrame)

# 2.1 EW Covariance (λ=0.97)
cout = ewCovar(Matrix(x), 0.97)
CSV.write("data/testout_2.1.csv", DataFrame(cout, :auto))

# 2.2 EW Correlation (λ=0.94)
cout = ewCovar(Matrix(x), 0.94)
sd = 1 ./ sqrt.(diag(cout))
cout = diagm(sd) * cout * diagm(sd)
CSV.write("data/testout_2.2.csv", DataFrame(cout, :auto))

# 2.3 Mixed EW Covariance/Correlation
# EW Covariance with λ=0.97, then rescale by EW variance (λ=0.94)
cout = ewCovar(Matrix(x), 0.97)
sd1 = sqrt.(diag(cout))        # EW Std (λ=0.97)
cout = ewCovar(Matrix(x), 0.94)
sd = 1 ./ sqrt.(diag(cout))    # EW Std (λ=0.94)
cout = diagm(sd1) * diagm(sd) * cout * diagm(sd) * diagm(sd1)
CSV.write("data/testout_2.3.csv", DataFrame(cout, :auto))
