include(joinpath(@__DIR__, "fit_general_t.jl"))

using CSV, DataFrames

# 7.2 Fit TDist
cin = CSV.read("data/test7_2.csv",DataFrame) |> Matrix
fd = fit_general_t(cin[:,1])
CSV.write("data/testout7_2.csv",DataFrame(:mu=>[fd.errorModel.μ],:sigma=>[fd.errorModel.σ],:nu=>[fd.errorModel.ρ.ν]))
