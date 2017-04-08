# Conjugate

"""
  Conjugate(f::ProximableFunction)

Returns the conjugate function of `f`, that is `f*(x) = sup{y'x - f(y)}`.
"""

immutable Conjugate{T <: ProximableFunction} <: ProximableFunction
  f::T
  function Conjugate(f::T)
    if is_convex(f) == false
      error("`f` must be convex")
    end
    new(f)
  end
end

is_prox_accurate(f::Conjugate) = is_prox_accurate(f.f)
is_convex(f::Conjugate) = true
is_cone(f::Conjugate) = is_cone(f.f) && is_convex(f.f)

fun_dom(f::Conjugate) = fun_dom(f.f)

Conjugate{T <: ProximableFunction}(f::T) = Conjugate{T}(f)

# only prox! is provided here, call method would require being able to compute
# an element of the subdifferential of the conjugate

function prox!{R <: Real}(y::AbstractArray{R}, g::Conjugate, x::AbstractArray{R}, gamma::Real=1.0)
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

function prox!{R <: Real}(y::AbstractArray{Complex{R}}, g::Conjugate, x::AbstractArray{Complex{R}}, gamma::Real=1.0)
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

function prox_naive{T <: RealOrComplex}(g::Conjugate, x::AbstractArray{T}, gamma::Real=1.0)
  y, v = prox_naive(g.f, x/gamma, 1.0/gamma)
  return x - gamma*y, real(vecdot(x,y)) - gamma*real(vecdot(y,y)) - v
end
