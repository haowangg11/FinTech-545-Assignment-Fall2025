using CSV, DataFrames
include("fit_general_t3.jl")

# Test 8.3 VaR Simulation
cin = CSV.read("data/test7_2.csv",DataFrame) |> Matrix
fd = fit_general_t(cin[:,1])
sim = fd.eval(rand(10000))
CSV.write("data/testout8_3.csv",
    DataFrame(Symbol("VaR Absolute")=>[VaR(sim)],
            Symbol("VaR Diff from Mean")=>[VaR(sim .- mean(sim))]
))
