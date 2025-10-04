using CSV, DataFrames, Distributions
include("fit_normal.jl")

# Test 8.1 VaR Normal
cin = CSV.read("data/test7_1.csv",DataFrame) |> Matrix
fd = fit_normal(cin[:,1])
CSV.write("data/testout8_1.csv",
    DataFrame(Symbol("VaR Absolute")=>[VaR(fd.errorModel)],
            Symbol("VaR Diff from Mean")=>[-quantile(Normal(0,fd.errorModel.σ),0.05)]
))
