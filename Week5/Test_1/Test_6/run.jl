using CSV, DataFrames
include("return_calculate.jl")

# Test 6

# 6.1 Arithmetic returns
prices = CSV.read("data/test6.csv",DataFrame)
rout = return_calculate(prices,dateColumn="Date")
CSV.write("data/test6_1.csv",rout)

# 6.2 Log returns
prices = CSV.read("data/test6.csv",DataFrame)
rout = return_calculate(prices,method="LOG", dateColumn="Date")
CSV.write("data/test6_2.csv",rout)
