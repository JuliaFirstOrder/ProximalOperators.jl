# indicator of a simplex

export IndSimplex

"""
**Indicator of a simplex**

    IndSimplex(a=1.0)

Returns the indicator of the set
```math
S = \\left\\{ x : x \\geq 0, ∑_i x_i = a \\right\\}.
```
By default `a=1.0`, therefore ``S`` is the probability simplex.
"""
struct IndSimplex{R <: Real} <: ProximableFunction
    a::R
    function IndSimplex{R}(a::R) where {R <: Real}
        if a <= 0
            error("parameter a must be positive")
        else
            new(a)
        end
    end
end

is_convex(f::IndSimplex) = true
is_set(f::IndSimplex) = true

IndSimplex(a::R=1.0) where {R <: Real} = IndSimplex{R}(a)

function (f::IndSimplex{T})(x::AbstractArray{R}) where {T, R <: Real}
    if all(x .>= 0) && sum(x) ≈ f.a
        return zero(R)
    end
    return R(Inf)
end

function prox!(y::AbstractArray{R}, f::IndSimplex{T}, x::AbstractArray{R}, gamma::R=one(R)) where {T, R <: Real}
# Implements Algorithm 1 in Condat, "Fast projection onto the simplex and the l1 ball", Mathematical Programming, 158:575–585, 2016.
# We should consider implementing the other algorithms reviewed there, and the one proposed in the paper.
    n = length(x)
    p = []
    if ndims(x) == 1
        p = sort(x, rev=true)
    else
        p = sort(x[:], rev=true)
    end
    s = 0
    for i = 1:n-1
        s = s + p[i]
        tmax = (s - f.a)/i
        if tmax >= p[i+1]
            @inbounds for j in eachindex(y)
                y[j] = x[j] < tmax ? zero(R) : x[j] - tmax
            end
            return zero(R)
        end
    end
    tmax = (s + p[n] - f.a)/n
    @inbounds for j in eachindex(y)
        y[j] = x[j] < tmax ? zero(R) : x[j] - tmax
    end
    return zero(R)
end

fun_name(f::IndSimplex) = "indicator of the probability simplex"
fun_dom(f::IndSimplex) = "AbstractArray{Real}"
fun_expr(f::IndSimplex) = "x ↦ 0 if x ⩾ 0 and sum(x) = a, +∞ otherwise"
fun_params(f::IndSimplex) = "a = $(f.a)"

function prox_naive(f::IndSimplex{T}, x::AbstractArray{R}, gamma::R=one(R)) where {T, R <: Real}
    low = minimum(x)
    upp = maximum(x)
    v = x
    s = Inf
    for i = 1:100
        if abs(s)/f.a ≈ 0
            break
        end
        alpha = (low+upp)/2
        v = max.(x .- alpha, zero(R))
        s = sum(v) - f.a
        if s <= 0
            upp = alpha
        else
            low = alpha
        end
    end
    return v, zero(R)
end
