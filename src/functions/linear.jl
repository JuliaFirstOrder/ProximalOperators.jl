export Linear

"""
**Linear function**

    Linear(c)

Returns the function
```math
f(x) = \\langle c, x \\rangle.
```
"""
struct Linear{R <: Real, A <: AbstractArray{R}}
    c::A
end

is_separable(f::Linear) = true
is_convex(f::Linear) = true
is_smooth(f::Linear) = true

function (f::Linear{R})(x::AbstractArray{R}) where R
    return dot(f.c, x)
end

function prox!(y::AbstractArray{R}, f::Linear{R}, x::AbstractArray{R}, gamma::Union{R, AbstractArray{R}}=1.0) where R
    y .= x .- gamma.*(f.c)
    fy = dot(f.c, y)
    return fy
end

function gradient!(y::AbstractArray{R}, f::Linear{R}, x::AbstractArray{R}) where R
    y .= f.c
    return dot(f.c, x)
end

function prox_naive(f::Linear, x, gamma)
    y = x - gamma.*(f.c)
    fy = dot(f.c, y)
    return y, fy
end
