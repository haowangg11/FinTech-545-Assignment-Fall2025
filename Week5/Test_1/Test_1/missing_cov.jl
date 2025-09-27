using Statistics, DataFrames

function missing_cov(x; skipMiss=true, fun=cov)
    if skipMiss
        df = DataFrame(x, :auto)
        df_clean = dropmissing(df)
        return fun(Matrix(df_clean))
    else
        p = size(x, 2)
        C = Matrix{Float64}(undef, p, p)

        for i in 1:p
            for j in 1:p
                xi = x[:, i]
                xj = x[:, j]

                mask = .!(ismissing.(xi) .| ismissing.(xj))

                xi_clean = Float64.(xi[mask])
                xj_clean = Float64.(xj[mask])

                if length(xi_clean) > 1
                    C[i, j] = fun === cov ? cov(xi_clean, xj_clean) : cor(xi_clean, xj_clean)
                else
                    C[i, j] = NaN
                end
            end
        end

        return C
    end
end
