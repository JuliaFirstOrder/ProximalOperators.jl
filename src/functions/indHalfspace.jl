# indicator of a halfspace

export IndHalfspace

"""
    IndHalfspace(a, b)

For an array `a` and a scalar `b`, return the indicator of half-space
```math
S = \\{x : \\langle a,x \\rangle \\leq b \\}.
```
"""
struct IndHalfspace{R, T}
    a::T
    b::R
    norm_a::R
    function IndHalfspace{R, T}(a::T, b::R) where {R, T}
        norm_a = norm(a)
        if isapprox(norm_a, 0) && b < 0
            error("function is improper")
        end
        new(a, b, norm_a)
    end
end

IndHalfspace(a::T, b::R) where {R, T} = IndHalfspace{R, T}(a, b)

is_convex(f::Type{<:IndHalfspace}) = true
is_set_indicator(f::Type{<:IndHalfspace}) = true

function (f::IndHalfspace)(x)
    R = real(eltype(x))
    if isapprox_le(dot(f.a, x), f.b, atol=eps(R), rtol=sqrt(eps(R)))
        return R(0)
    end
    return R(Inf)
end

function prox!(y, f::IndHalfspace, x, gamma)
    R = real(eltype(x))
    s = dot(f.a, x)
    if s > f.b
        y .= x .- ((s - f.b)/f.norm_a^2) .* f.a
    else
        copyto!(y, x)
    end
    return R(0)
end

function prox_naive(f::IndHalfspace, x, gamma)
    R = real(eltype(x))
    s = dot(f.a, x) - f.b
    if s <= 0
        return x, 0.0
    end
    return x - (s/norm(f.a)^2)*f.a, R(0)
end
