include("fit_t_ES.jl")

# Test 8.5 ES TDist
cin = CSV.read("data/test7_2.csv",DataFrame) |> Matrix
fd = fit_general_t(cin[:,1])
CSV.write("data/testout8_5.csv",
    DataFrame(Symbol("ES Absolute")=>[ES(fd.errorModel)],
            Symbol("ES Diff from Mean")=>[ES(TDist(fd.errorModel.ρ.ν)*fd.errorModel.σ)]
))
