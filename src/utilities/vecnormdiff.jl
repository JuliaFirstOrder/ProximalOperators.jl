"""
Fast (non-allocating) version of vecnorm(x-y,2)^2
"""
function vecnormdiff2(x,y)
    s = 0.0
    for i in eachindex(x)
        s += abs2(x[i]-y[i])
    end
    return s
end

"""
Fast (non-allocating) version of vecnorm(x-y,2)
"""
vecnormdiff(x,y) = sqrt(vecnormdiff2(x,y))
