using CSV, DataFrames
include("simulate_pca.jl")

# 5.5 PSD Input - PCA Simulation
cin = CSV.read("data/test5_2.csv",DataFrame) |> Matrix
cout = cov(simulate_pca(cin,100000,pctExp=.99))
CSV.write("data/testout_5.5.csv",DataFrame(cout,:auto))