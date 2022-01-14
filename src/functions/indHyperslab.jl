# indicator of a hyperslab

export IndHyperslab

"""
**Indicator of a hyperslab**

    IndHyperslab(low, a, upp)

For an array `a` and scalars `low` and `upp`, returns the indicator of set
```math
S = \\{x : low \\leq \\langle a,x \\rangle \\leq upp \\}.
```
"""
struct IndHyperslab{R <: Real, T <: AbstractArray{R}} <: ProximableFunction
    low::R
    a::T
    upp::R
    norm_a::R
    function IndHyperslab{R, T}(low::R, a::T, upp::R) where {R <: Real, T <: AbstractArray{R}}
        norm_a = norm(a)
        if (norm_a == 0 && (upp < 0 || low > 0)) || upp < low
            error("function is improper")
        end
        new(low, a, upp, norm_a)
    end
end

IndHyperslab(low::R, a::T, upp::R) where {R <: Real, T <: AbstractArray{R}} = IndHyperslab{R, T}(low, a, upp)

is_convex(f::IndHyperslab) = true
is_set(f::IndHyperslab) = true
is_cone(f::IndHyperslab{R}) where R =
    iszero(f.norm_a) ||
    (f.low == f.upp == 0) ||
    (f.low == 0 && f.upp == Inf) ||
    (f.low == -Inf && f.upp == 0) ||
    (f.low == -Inf && f.upp == Inf)

function (f::IndHyperslab{R})(x::AbstractArray{R}) where R
    if iszero(f.norm_a)
        return R(0)
    end
    s = dot(f.a, x)
    tol = eps(R) * f.norm_a
    if isapprox_le(f.low, s, atol=tol, rtol=tol) && isapprox_ge(f.upp, s, atol=tol, rtol=tol)
        return R(0)
    end
    return R(Inf)
end

function prox!(y::AbstractArray{R}, f::IndHyperslab{R}, x::AbstractArray{R}, gamma::R=R(1)) where R
    s = dot(f.a, x)
    if s < f.low && f.norm_a > 0
        y .= x .- ((s - f.low)/f.norm_a^2) .* f.a
    elseif s > f.upp && f.norm_a > 0
        y .= x .- ((s - f.upp)/f.norm_a^2) .* f.a
    else
        copyto!(y, x)
    end
    return R(0)
end

function prox_naive(f::IndHyperslab{R}, x::AbstractArray{R}, gamma::R=R(1)) where R
    s = dot(f.a, x)
    if s < f.low && f.norm_a > 0
        return x - ((s - f.low)/norm(f.a)^2) * f.a, R(0)
    elseif s > f.upp && f.norm_a > 0
        return x - ((s - f.upp)/norm(f.a)^2) * f.a, R(0)
    else
        return x, R(0)
    end
end
