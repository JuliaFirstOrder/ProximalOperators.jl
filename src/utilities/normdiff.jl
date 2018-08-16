"""
Fast (non-allocating) version of norm(x-y,2)^2
"""
function normdiff2(x,y)
    s = 0.0
    for i in eachindex(x)
        s += abs2(x[i]-y[i])
    end
    return s
end

"""
Fast (non-allocating) version of norm(x-y,2)
"""
normdiff(x,y) = sqrt(normdiff2(x,y))
