using DataFrames, Statistics, Dates

function return_calculate(df::DataFrame; method::String="ARITH", dateColumn::String="")
    cols = names(df)
    if dateColumn != ""
        cols = filter(c -> c != dateColumn, cols)
    end

    rets = DataFrame()
    if dateColumn != ""
        rets[!, dateColumn] = df[2:end, dateColumn]
    end

    for c in cols
        prices = df[!, c]
        if method == "LOG"
            rets[!, c] = log.(prices[2:end] ./ prices[1:end-1])
        else
            rets[!, c] = prices[2:end] ./ prices[1:end-1] .- 1
        end
    end
    return rets
end
