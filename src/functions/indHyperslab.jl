# indicator of a hyperslab

export IndHyperslab

"""
    IndHyperslab(low, a, upp)

For an array `a` and scalars `low` and `upp`, return the so-called hyperslab
```math
S = \\{x : low \\leq \\langle a,x \\rangle \\leq upp \\}.
```
"""
struct IndHyperslab{R, T}
    low::R
    a::T
    upp::R
    norm_a::R
    function IndHyperslab{R, T}(low::R, a::T, upp::R) where {R, T}
        norm_a = norm(a)
        if (norm_a == 0 && (upp < 0 || low > 0)) || upp < low
            error("function is improper")
        end
        new(low, a, upp, norm_a)
    end
end

IndHyperslab(low::R, a::T, upp::R) where {R, T} = IndHyperslab{R, T}(low, a, upp)

is_convex(f::Type{<:IndHyperslab}) = true
is_set(f::Type{<:IndHyperslab}) = true

function (f::IndHyperslab)(x)
    R = real(eltype(x))
    if iszero(f.norm_a)
        return R(0)
    end
    s = dot(f.a, x)
    tol = 100 * eps(R) * f.norm_a
    if isapprox_le(f.low, s, atol=tol, rtol=tol) && isapprox_le(s, f.upp, atol=tol, rtol=tol)
        return R(0)
    end
    return R(Inf)
end

function prox!(y, f::IndHyperslab, x, gamma)
    s = dot(f.a, x)
    if s < f.low && f.norm_a > 0
        y .= x .- ((s - f.low)/f.norm_a^2) .* f.a
    elseif s > f.upp && f.norm_a > 0
        y .= x .- ((s - f.upp)/f.norm_a^2) .* f.a
    else
        copyto!(y, x)
    end
    return real(eltype(x))(0)
end

function prox_naive(f::IndHyperslab, x, gamma)
    R = real(eltype(x))
    s = dot(f.a, x)
    if s < f.low && f.norm_a > 0
        return x - ((s - f.low)/norm(f.a)^2) * f.a, R(0)
    elseif s > f.upp && f.norm_a > 0
        return x - ((s - f.upp)/norm(f.a)^2) * f.a, R(0)
    else
        return x, R(0)
    end
end
