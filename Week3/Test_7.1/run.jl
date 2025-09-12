include(joinpath(@__DIR__, "fit_normal.jl"))

using CSV, DataFrames

# 7.1 Fit Normal Distribution
cin = CSV.read("data/test7_1.csv",DataFrame) |> Matrix
fd = fit_normal(cin[:,1])

@show typeof(fd)
@show fd

CSV.write("data/testout7_1.csv",DataFrame(:mu=>[fd.errorModel.μ],:sigma=>[fd.errorModel.σ]))


