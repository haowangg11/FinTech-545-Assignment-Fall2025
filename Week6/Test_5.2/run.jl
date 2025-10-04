using CSV, DataFrames, Statistics
include("simulateNormal2.jl")

# 5.2 PSD Input
cin = CSV.read("data/test5_2.csv",DataFrame) |> Matrix
cout = cov(simulateNormal(100000, cin))
CSV.write("data/testout_5.2.csv",DataFrame(cout,:auto))