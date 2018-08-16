export Linear

"""
**Linear function**

    Linear(c)

Returns the function
```math
f(x) = \\langle c, x \\rangle.
```
"""

struct Linear{R <: RealOrComplex, A <: AbstractArray{R}} <: ProximableFunction
  c::A
end

is_separable(f::Linear) = true
is_convex(f::Linear) = true
is_smooth(f::Linear) = true

function (f::Linear{RC, A})(x::AbstractArray{RC}) where {R <: Real, RC <: Union{R, Complex{R}}, A <: AbstractArray{RC}}
  return dot(f.c, x)
end

function prox!(y::AbstractArray{RC}, f::Linear{RC, A}, x::AbstractArray{RC}, gamma::Union{R, AbstractArray{R}}=1.0) where {R <: Real, RC <: Union{R, Complex{R}}, A <: AbstractArray{RC}}
  y .= x .- gamma.*(f.c)
  fy = dot(f.c, y)
  return fy
end

function gradient!(y::AbstractArray{RC}, f::Linear{RC, A}, x::AbstractArray{RC}) where {R <: Real, RC <: Union{R, Complex{R}}, A <: AbstractArray{RC}}
  y .= f.c
  return dot(f.c, x)
end

fun_name(f::Linear) = "Linear function"
fun_expr(f::Linear) = "x ↦ c'x"

function prox_naive(f::Linear, x, gamma)
  y = x - gamma.*(f.c)
  fy = dot(f.c, y)
  return y, fy
end
