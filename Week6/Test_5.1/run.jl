using CSV, DataFrames, Statistics

include("simulateNormal.jl")

cin = CSV.read("data/test5_1.csv", DataFrame) |> Matrix
cout = cov(simulateNormal(100000, cin))
CSV.write("data/testout_5.1.csv", DataFrame(cout, :auto))