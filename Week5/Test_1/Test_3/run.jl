using CSV, DataFrames, LinearAlgebra
include("near_psd.jl")
include("higham_psd.jl")

#Test 3 - non-psd matrices

#3.1 near_psd covariance
cin = CSV.read("data/testout_1.3.csv",DataFrame)
cout = near_psd(Matrix(cin))
CSV.write("data/testout_3.1.csv",DataFrame(cout,:auto))

#3.2 near_psd Correlation
cin = CSV.read("data/testout_1.4.csv",DataFrame)
cout = near_psd(Matrix(cin))
CSV.write("data/testout_3.2.csv",DataFrame(cout,:auto))

#3.3 Higham covariance
cin = CSV.read("data/testout_1.3.csv",DataFrame)
cout = higham_nearestPSD(Matrix(cin))
CSV.write("data/testout_3.3.csv",DataFrame(cout,:auto))

#3.4 Higham Correlation
cin = CSV.read("data/testout_1.4.csv",DataFrame)
cout = higham_nearestPSD(Matrix(cin))
CSV.write("data/testout_3.4.csv",DataFrame(cout,:auto))
