using CSV, DataFrames
include("simulateNormal3.jl")

cin = CSV.read("data/test5_3.csv", DataFrame) |> Matrix
cout = cov(simulateNormal(100000, cin; fixMethod=:near_psd))
CSV.write("data/testout_5.3.csv", DataFrame(cout, :auto))
