# Conjugate

export Conjugate

"""
**Convex conjugate**

    Conjugate(f)

Returns the convex conjugate (also known as Fenchel conjugate, or Fenchel-Legendre transform) of function `f`, that is
```math
f^*(x) = \\sup_y \\{ \\langle y, x \\rangle - f(y) \\}.
```
"""

struct Conjugate{T <: ProximableFunction} <: ProximableFunction
  f::T
  function Conjugate{T}(f::T) where {T<: ProximableFunction}
    if is_convex(f) == false
      error("`f` must be convex")
    end
    new(f)
  end
end

is_prox_accurate(f::Conjugate) = is_prox_accurate(f.f)
is_convex(f::Conjugate) = true
is_cone(f::Conjugate) = is_cone(f.f) && is_convex(f.f)
is_smooth(f::Conjugate) = is_strongly_convex(f.f)
is_strongly_convex(f::Conjugate) = is_smooth(f.f)
is_quadratic(f::Conjugate) = is_strongly_convex(f.f) && is_generalized_quadratic(f.f)
is_generalized_quadratic(f::Conjugate) = is_quadratic(f.f)

fun_dom(f::Conjugate) = fun_dom(f.f)

Conjugate(f::T) where {T <: ProximableFunction} = Conjugate{T}(f)

# only prox! is provided here, call method would require being able to compute
# an element of the subdifferential of the conjugate

function prox!(y::AbstractArray{R}, g::Conjugate, x::AbstractArray{R}, gamma::Real=1.0) where R <: Real
  # Moreau identity
  v = prox!(y, g.f, x/gamma, 1.0/gamma)
  if is_set(g)
    v = 0.0
  else
    v = vecdot(x,y) - gamma*vecdot(y,y) - v
  end
  for k in eachindex(y)
    y[k] = x[k] - gamma*y[k]
  end
  return v
end

# complex case, need to cast inner products to real

function prox!(y::AbstractArray{Complex{R}}, g::Conjugate, x::AbstractArray{Complex{R}}, gamma::Real=1.0) where R <: Real
  v = prox!(y, g.f, x/gamma, 1.0/gamma)
  if is_set(g)
    v = 0.0
  else
    v = real(vecdot(x,y)) - gamma*real(vecdot(y,y)) - v
  end
  for k in eachindex(y)
    y[k] = x[k] - gamma*y[k]
  end
  return v
end

# naive implementation

function prox_naive(g::Conjugate, x::AbstractArray{T}, gamma::Real=1.0) where T <: RealOrComplex
  y, v = prox_naive(g.f, x/gamma, 1.0/gamma)
  return x - gamma*y, real(vecdot(x,y)) - gamma*real(vecdot(y,y)) - v
end
