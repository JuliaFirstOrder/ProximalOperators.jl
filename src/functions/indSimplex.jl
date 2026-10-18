# indicator of a simplex

export IndSimplex, IndUnitSimplex

"""
    IndSimplex(a=1.0)

Return the indicator of the simplex
```math
S = \\left\\{ x : x \\geq 0, \\sum_i x_i = a \\right\\}.
```

By default `a=1`, therefore ``S`` is the probability simplex.
"""
struct IndSimplex{R}
    a::R
    function IndSimplex{R}(a::R) where R
        if a <= 0
            error("parameter a must be positive")
        else
            new(a)
        end
    end
end

is_convex(f::Type{<:IndSimplex}) = true
is_set_indicator(f::Type{<:IndSimplex}) = true

IndSimplex(a::R=1) where R = IndSimplex{R}(a)

function (f::IndSimplex)(x)
    R = eltype(x)
    if all(x .>= 0) && sum(x) ≈ f.a
        return R(0)
    end
    return R(Inf)
end

function simplex_proj_condat!(y, a, x)
    # Implements algorithm proposed in:
    # Condat, L. "Fast projection onto the simplex and the l1 ball",
    # Mathematical Programming, 158:575–585, 2016.
    R = eltype(x)
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
    v_changed = true
    while v_changed == true
        v_changed = false
        k = 1
        while k <= length(v)
            z = v[k]
            if z <= rho
                deleteat!(v, k)
                v_changed = true
                rho += (rho - z) / length(v)
            else
                k = k + 1
            end
        end
    end
    y .= max.(x .- rho, R(0))
end

function prox!(y, f::IndSimplex, x, gamma)
    simplex_proj_condat!(y, f.a, x)
    return eltype(x)(0)
end

function prox_naive(f::IndSimplex, x, gamma)
    R = eltype(x)
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

"""
    IndUnitSimplex(a=1.0)

Return the indicator of the unit simplex
```math
S = \\left\\{ x : x \\geq 0, \\sum_i x_i \\leq a \\right\\}.
```

By default `a=1`, therefore ``S`` is the probability simplex of dimension n+1.
"""
struct IndUnitSimplex{R}
    a::R
    function IndUnitSimplex{R}(a::R) where R
        if a <= 0
            error("parameter a must be positive")
        else
            new(a)
        end
    end
end

is_convex(f::Type{<:IndUnitSimplex}) = true
is_set(f::Type{<:IndUnitSimplex}) = true

IndUnitSimplex(a::R=1) where R = IndUnitSimplex{R}(a)

function (f::IndUnitSimplex)(x)
    R = eltype(x)
    if all(x .>= 0) && sum(x) <= f.a + eps(f.a)
        return R(0)
    end
    return R(Inf)
end

function prox!(y, f::IndUnitSimplex{R}, x, gamma) where {R}
    fx = zero(R)
    for i in eachindex(x)
        y[i] = max(x[i], zero(R))
        fx += y[i]
    end
    if fx > f.a
        simplex_proj_condat!(y, f.a, x)
    end
    return eltype(x)(0)
end
