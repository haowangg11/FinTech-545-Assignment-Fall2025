include("chol_psd!.jl")

#4 cholesky factorization
cin = Matrix(CSV.read("data/testout_3.1.csv",DataFrame))
n,m = size(cin)
cout = zeros(Float64,(n,m))
chol_psd!(cout,cin)
CSV.write("data/testout_4.1.csv",DataFrame(cout,:auto))
