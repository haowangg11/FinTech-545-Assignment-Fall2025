include("fit_normal_ES.jl")

cin = CSV.read("data/test7_1.csv", DataFrame) |> Matrix
fd = fit_normal(cin[:,1])
CSV.write("data/testout8_4.csv",
    DataFrame(Symbol("ES Absolute") => [ES(fd.errorModel)],
              Symbol("ES Diff from Mean") => [ES(Normal(0, fd.errorModel.σ))])
)
