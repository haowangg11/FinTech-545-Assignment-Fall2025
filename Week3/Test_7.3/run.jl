include(joinpath(@__DIR__, "fit_regression_t.jl"))

using CSV, DataFrames

# 7.3 Fit T Regression
cin = CSV.read("data/test7_3.csv",DataFrame)
fd = fit_regression_t(cin.y,Matrix(select(cin,Not(:y))))
CSV.write("data/testout7_3.csv",
    DataFrame(:mu=>[fd.errorModel.μ],
            :sigma=>[fd.errorModel.σ],
            :nu=>[fd.errorModel.ρ.ν],
            :Alpha=>[fd.beta[1]],
            :B1=>[fd.beta[2]],
            :B2=>[fd.beta[3]],
            :B3=>[fd.beta[4]]   
))