include(joinpath(@__DIR__, "fit_normal.jl"))

using CSV, DataFrames

cin = CSV.read("data/test7_1.csv",DataFrame) |> Matrix
fd = fit_normal(cin[:,1])
CSV.write("data/testout7_1.csv",DataFrame(    
    :mu => [fd.μ],
    :sigma => [fd.σ]))

