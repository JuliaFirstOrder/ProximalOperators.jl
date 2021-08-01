# indicator of a simplex

export IndSimplex

"""
**Indicator of a simplex**

    IndSimplex(a=1.0)

Returns the indicator of the set
```math
S = \\left\\{ x : x \\geq 0, \\sum_i x_i = a \\right\\}.
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

function (f::IndSimplex)(x::AbstractArray{R}) where R <: Real
    if all(x .>= 0) && sum(x) ≈ f.a
        return R(0)
    end
    return R(Inf)
end

function simplex_proj_condat!(y::AbstractArray{R}, a, x::AbstractArray{R}) where R
    # Implements algorithm proposed in:
    # Condat, L. "Fast projection onto the simplex and the l1 ball",
    # Mathematical Programming, 158:575–585, 2016.
    v = [x[1]]
    v_tilde = R[]
    rho = x[1] - a
    N = length(x)
    for k in 2:N
        if x[k] > rho
            rho += (x[k] - rho) / (length(v) + 1)
            if rho > x[k] - a
                push!(v, x[k])
            else
                append!(v_tilde, v)
                v = [x[k]]
                rho = x[k] - a
            end
        end
    end
    for z in v_tilde
        if z > rho
            push!(v, z)
            rho += (z - rho) / length(v)
        end
    end
    v_changes = true
    while v_changes == true
        v_changes = false
        k = 1
        while k <= length(v)
            z = v[k]
            if z <= rho
                popat!(v, k)
                v_changes = true
                rho += (rho - z) / length(v)
            else
                k = k + 1
            end
        end
    end
    y .= max.(x .- rho, R(0))
end

function prox!(y::AbstractArray{R}, f::IndSimplex, x::AbstractArray{R}, _::R=R(1)) where R <: Real
    simplex_proj_condat!(y, f.a, x)
    return R(0)
end

fun_name(f::IndSimplex) = "indicator of the probability simplex"
fun_dom(f::IndSimplex) = "AbstractArray{Real}"
fun_expr(f::IndSimplex) = "x ↦ 0 if x ⩾ 0 and sum(x) = a, +∞ otherwise"
fun_params(f::IndSimplex) = "a = $(f.a)"

function prox_naive(f::IndSimplex, x::AbstractArray{R}, _::R=R(1)) where R <: Real
    low = minimum(x)
    upp = maximum(x)
    v = x
    s = Inf
    for i = 1:100
        if abs(s)/f.a ≈ 0
            break
        end
        alpha = (low+upp)/2
        v = max.(x .- alpha, R(0))
        s = sum(v) - f.a
        if s <= 0
            upp = alpha
        else
            low = alpha
        end
    end
    return v, R(0)
end
