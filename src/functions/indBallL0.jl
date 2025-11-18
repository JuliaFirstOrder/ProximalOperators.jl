# indicator of the L0 norm ball with given (integer) radius

export IndBallL0

"""
    IndBallL0(r=1)

Return the indicator function of the ``L_0`` pseudo-norm ball
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

is_set_indicator(f::Type{<:IndBallL0}) = true

IndBallL0(r::I) where {I} = IndBallL0{I}(r)

function (f::IndBallL0)(x)
    R = real(eltype(x))
    if count(!isequal(0), x) > f.r
        return R(Inf)
    end
    return R(0)
end

function _get_top_k_abs_indices(x::AbstractVector, k)
    range = firstindex(x):(firstindex(x) + k - 1)
    return partialsortperm(x, range, by=abs, rev=true)
end

_get_top_k_abs_indices(x, k) = _get_top_k_abs_indices(x[:], k)

function prox!(y, f::IndBallL0, x, gamma)
    T = eltype(x)
    p = _get_top_k_abs_indices(x, f.r)
    y .= T(0)
    for i in eachindex(p)
        y[p[i]] = x[p[i]]
    end
    return real(T)(0)
end

function prox_naive(f::IndBallL0, x, gamma)
    T = eltype(x)
    p = sortperm(abs.(x)[:], rev=true)
    y = similar(x)
    y[p[begin:begin+f.r-1]] .= x[p[begin:begin+f.r-1]]
    y[p[begin+f.r:end]] .= T(0)
    return y, real(T)(0)
end
