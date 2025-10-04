using CSV, DataFrames, Distributions
include("fit_general_t.jl")

# Test 8.2 VaR TDist
cin = CSV.read("data/test7_2.csv",DataFrame) |> Matrix
fd = fit_general_t(cin[:,1])
CSV.write("data/testout8_2.csv",
    DataFrame(Symbol("VaR Absolute")=>[VaR(fd.errorModel)],
            Symbol("VaR Diff from Mean")=>[-quantile(TDist(fd.errorModel.ρ.ν)*fd.errorModel.σ,0.05)]
))