using LinearAlgebra

function chol_psd!(L::Matrix{Float64}, A::Matrix{Float64})
    n = size(A, 1)
    @assert n == size(A, 2) 
    @assert size(L) == size(A) 

    eps_tol = 1e-10  
    L .= 0.0         

    for i in 1:n
        temp = A[i,i] - sum(L[i,1:i-1].^2)
        L[i,i] = sqrt(max(temp, eps_tol))  
        for j in i+1:n
            L[j,i] = (A[j,i] - sum(L[j,1:i-1] .* L[i,1:i-1])) / L[i,i]
        end
    end
    return L
end
