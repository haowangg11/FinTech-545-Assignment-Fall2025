include("missing_cov.jl")
x = CSV.read("data/test1.csv",DataFrame)

#1.1 Skip Missing rows - Covariance
cout = missing_cov(Matrix(x),skipMiss=true)
CSV.write("data/testout_1.1.csv",DataFrame(cout,:auto))
#1.2 Skip Missing rows - Correlation
cout = missing_cov(Matrix(x),skipMiss=true,fun=cor)
CSV.write("data/testout_1.2.csv",DataFrame(cout,:auto))
#1.3 Pairwise - Covariance
cout = missing_cov(Matrix(x),skipMiss=false)
CSV.write("data/testout_1.3.csv",DataFrame(cout,:auto))
#1.2 Pairwise - Correlation
cout = missing_cov(Matrix(x),skipMiss=false,fun=cor)
CSV.write("data/testout_1.4.csv",DataFrame(cout,:auto))