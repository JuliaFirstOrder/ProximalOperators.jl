export Linear

"""
    Linear(c)

Return the linear function
```math
f(x) = \\langle c, x \\rangle.
```
"""
struct Linear{A}
    c::A
end

is_separable(f::Type{<:Linear}) = true
is_convex(f::Type{<:Linear}) = true
is_smooth(f::Type{<:Linear}) = true

function (f::Linear)(x)
    return dot(f.c, x)
end

function prox!(y, f::Linear, x, gamma)
    y .= x .- gamma .* f.c
    fy = dot(f.c, y)
    return fy
end

function gradient!(y, f::Linear, x)
    y .= f.c
    return dot(f.c, x)
end

function prox_naive(f::Linear, x, gamma)
    y = x - gamma .* f.c
    fy = dot(f.c, y)
    return y, fy
end
