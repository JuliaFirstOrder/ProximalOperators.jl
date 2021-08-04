# indicator of the L0 norm ball with given (integer) radius

export IndBallL0

"""
**Indicator of a ``L_0`` pseudo-norm ball**

    IndBallL0(r=1)

Returns the indicator function of the set
```math
S = \\{ x : \\mathrm{nnz}(x) \\leq r \\}.
```
Parameter `r` must be a positive integer.
"""
struct IndBallL0{I}
    r::I
    function IndBallL0{I}(r::I) where {I}
        if r <= 0
            error("parameter r must be a positive integer")
        else
            new(r)
        end
    end
end

is_set(f::IndBallL0) = true

IndBallL0(r::I) where {I} = IndBallL0{I}(r)

function (f::IndBallL0)(x)
    R = real(eltype(x))
    if count(!isequal(0), x) > f.r
        return R(Inf)
    end
    return R(0)
end

function prox!(y, f::IndBallL0, x, gamma)
    T = eltype(x)
    p = []
    if ndims(x) == 1
        p = partialsortperm(x, 1:f.r, by=abs, rev=true)
    else
        p = partialsortperm(x[:], 1:f.r, by=abs, rev=true)
    end
    sort!(p)
    idx = 1
    for i = 1:length(p)
        y[idx:p[i]-1] .= T(0)
        y[p[i]] = x[p[i]]
        idx = p[i]+1
    end
    y[idx:end] .= T(0)
    return real(T)(0)
end

function prox_naive(f::IndBallL0, x, gamma)
    T = eltype(x)
    p = sortperm(abs.(x)[:], rev=true)
    y = similar(x)
    y[p[1:f.r]] .= x[p[1:f.r]]
    y[p[f.r+1:end]] .= T(0)
    return y, real(T)(0)
end
