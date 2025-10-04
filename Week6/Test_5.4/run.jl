using CSV, DataFrames
include("simulateNormal4.jl")
# 5.4 nonPSD Input Higham Fix
cin = CSV.read("data/test5_3.csv",DataFrame) |> Matrix
cout = cov(simulateNormal(100000, cin,fixMethod=higham_nearestPSD))
CSV.write("data/testout_5.4.csv",DataFrame(cout,:auto))