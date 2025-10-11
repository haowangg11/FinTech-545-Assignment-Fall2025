include("fit_general_t_sim.jl")

# Test 8.6 VaR Simulation
cin = CSV.read("data/test7_2.csv",DataFrame) |> Matrix
fd = fit_general_t(cin[:,1])
sim = fd.eval(rand(10000))
CSV.write("data/testout8_6.csv",
    DataFrame(Symbol("ES Absolute")=>[ES(sim)],
            Symbol("ES Diff from Mean")=>[ES(sim .- mean(sim))]
))