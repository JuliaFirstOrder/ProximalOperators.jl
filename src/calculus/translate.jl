export Translate

"""
**Translation**

    Translate(f, b)

Returns the translated function
```math
g(x) = f(x + b)
```
"""

immutable Translate{T <: ProximableFunction, V <: Union{Number, AbstractArray, Tuple}} <: ProximableFunction
  f::T
  b::V
end

is_separable(f::Translate) = is_separable(f.f)
is_prox_accurate(f::Translate) = is_prox_accurate(f.f)
is_convex(f::Translate) = is_convex(f.f)
is_set(f::Translate) = is_set(f.f)
is_singleton(f::Translate) = is_singleton(f.f)
is_cone(f::Translate) = is_cone(f.f)
is_affine(f::Translate) = is_affine(f.f)
is_smooth(f::Translate) = is_smooth(f.f)
is_quadratic(f::Translate) = is_quadratic(f.f)
is_generalized_quadratic(f::Translate) = is_generalized_quadratic(f.f)
is_strongly_convex(f::Translate) = is_strongly_convex(f.f)

function (g::Translate)(x::T) where {T <: Union{Tuple, AbstractArray}}
  return g.f(x .+ g.b)
end

function gradient!(y, g::Translate, x)
  z = x .+ g.b
  v = gradient!(y, g.f, z)
  return v
end

function prox!(y, g::Translate, x, gamma=1.0)
  z = x .+ g.b
  v = prox!(y, g.f, z, gamma)
  y .-= g.b
  return v
end

function prox_naive(g::Translate, x, gamma=1.0)
  y, v = prox_naive(g.f, x + g.b, gamma)
  return y - g.b, v
end

fun_name(f::Translate) = "Translation"
fun_dom(f::Translate) = fun_dom(f.f)
fun_expr(f::Translate) = "x ↦ f(x + b)"
