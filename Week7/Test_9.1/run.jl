include("risk_functions.jl")

# Test 9
# A = rand(Normal(0,.03),200)
# B = 0.1*A + rand(TDist(10)*.02,200)
# CSV.write("data/test9_1_returns.csv",DataFrame(:A=>A,:B=>B))

# 9.1
cin = CSV.read("data/test9_1_returns.csv",DataFrame)
prices = Dict{String,Float64}()
prices["A"] = 20.0
prices["B"] = 30

models = Dict{String,FittedModel}()
models["A"] = fit_normal(cin.A)
models["B"] = fit_general_t(cin.B)

nSim = 100000

U = [models["A"].u models["B"].u]
spcor = corspearman(U)
uSim = simulate_pca(spcor,nSim)
uSim = cdf.(Normal(),uSim)

simRet = DataFrame(:A=>models["A"].eval(uSim[:,1]), :B=>models["B"].eval(uSim[:,2]))

portfolio = DataFrame(:Stock=>["A","B"], :currentValue=>[2000.0, 3000.0])
iteration = [i for i in 1:nSim]
values = crossjoin(portfolio, DataFrame(:iteration=>iteration))

nv = size(values,1)
pnl = Vector{Float64}(undef,nv)
simulatedValue = copy(pnl)
for i in 1:nv
    simulatedValue[i] = values.currentValue[i] * (1 + simRet[values.iteration[i],values.Stock[i]])
    pnl[i] = simulatedValue[i] - values.currentValue[i]
end

values[!,:pnl] = pnl
values[!,:simulatedValue] = simulatedValue

risk = select(aggRisk(values,[:Stock]),[:Stock, :VaR95, :ES95, :VaR95_Pct, :ES95_Pct])

CSV.write("data/testout9_1.csv",risk)