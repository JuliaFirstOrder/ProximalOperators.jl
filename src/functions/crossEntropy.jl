# Cross Entropy loss function

export CrossEntropy

"""
    CrossEntropy(b)

Return the function
```math
f(x) = -\\frac{1}{N} \\sum_{i = 1}^{N} b_i \\log (x_i)+(1-b_i) \\log (1-x_i),
```
where `b` is an array of length `N` such that `0 ≤ b ≤ 1` component-wise.
"""
struct CrossEntropy{T}
    b::T
    function CrossEntropy{T}(b::T) where T
        if !(all(0 .<= b .<= 1. ))
            error("b must be 0 ≤ b ≤ 1 ")
        else
            new(b)
        end
    end
end

is_convex(f::Type{<:CrossEntropy}) = true
is_smooth(f::Type{<:CrossEntropy}) = true

CrossEntropy(b::T) where {T} = CrossEntropy{T}(b)

function (f::CrossEntropy)(x)
    fsum = eltype(x)(0)
    for i in eachindex(f.b)
        fsum += f.b[i]*log(x[i])+(1-f.b[i])*log(1-x[i])
    end
    return -fsum/length(f.b)
end

function gradient!(y, f::CrossEntropy, x)
    fsum = eltype(x)(0)
    for i in eachindex(x)
        y[i] = 1/length(f.b)*( - f.b[i]/x[i] + (1-f.b[i])/(1-x[i]) )
        fsum += f.b[i]*log(x[i])+(1-f.b[i])*log(1-x[i])
    end
    return -1/length(f.b)*fsum
end
