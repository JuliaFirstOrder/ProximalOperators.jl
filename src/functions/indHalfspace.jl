# indicator of a halfspace

export IndHalfspace

"""
**Indicator of a halfspace**

    IndHalfspace(a, b)

For an array `a` and a scalar `b`, returns the indicator of set
```math
S = \\{x : \\langle a,x \\rangle \\leq b \\}.
```
"""
struct IndHalfspace{R <: Real, T <: AbstractArray{R}}
    a::T
    b::R
    norm_a::R
    function IndHalfspace{R, T}(a::T, b::R) where {R <: Real, T <: AbstractArray{R}}
        norm_a = norm(a)
        if isapprox(norm_a, 0) && b < 0
            error("function is improper")
        end
        new(a, b, norm_a)
    end
end

IndHalfspace(a::T, b::R) where {R <: Real, T <: AbstractArray{R}} = IndHalfspace{R, T}(a, b)

is_convex(f::IndHalfspace) = true
is_set(f::IndHalfspace) = true
is_cone(f::IndHalfspace) = f.b == 0 || f.b == Inf

function (f::IndHalfspace{R})(x::AbstractArray{R}) where R
    if isapprox_le(dot(f.a, x), f.b, atol=eps(R), rtol=sqrt(eps(R)))
        return R(0)
    end
    return R(Inf)
end

function prox!(y::AbstractArray{R}, f::IndHalfspace{R}, x::AbstractArray{R}, gamma) where R
    s = dot(f.a, x)
    if s > f.b
        y .= x .- ((s - f.b)/f.norm_a^2) .* f.a
    else
        copyto!(y, x)
    end
    return R(0)
end

function prox_naive(f::IndHalfspace{R}, x::AbstractArray{R}, gamma) where R
    s = dot(f.a, x) - f.b
    if s <= 0
        return x, 0.0
    end
    return x - (s/norm(f.a)^2)*f.a, 0.0
end
