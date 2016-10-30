# Conjugate

immutable Conjugate{T <: ProximableFunction} <: ProximableFunction
  f::T
end

# only prox! is provided here, call method would require being able to compute
# an element of the subdifferential of the conjugate

function prox!{T <: RealOrComplex}(g::Conjugate, x::AbstractArray{T}, y::AbstractArray{T}, gamma::Real=1.0)
  # need to make a copy, doing this in place is probably not possible in general:
  # if object_id(x) == object_id(y), then prox! causes the loss of x (which is needed afterwards)
  x_copy = copy(x)
  # Moreau identity
  v = prox!(g.f, x/gamma, y, 1.0/gamma)
  v = vecdot(x_copy,y) - gamma*vecdot(y,y) - v
  y[:] *= -gamma
  y[:] += x_copy
  return v
end

# special case for indicator of convex cones (the conjugate is also an indicator function)

function prox!{T <: RealOrComplex, C <: IndicatorConvexCone}(g::Conjugate{C}, x::AbstractArray{T}, y::AbstractArray{T}, gamma::Real=1.0)
  x_copy = copy(x)
  prox!(g.f, x/gamma, y, 1.0/gamma)
  y[:] *= -gamma
  y[:] += x_copy
  return 0.0
end

function prox_naive{T <: RealOrComplex}(g::Conjugate, x::AbstractArray{T}, gamma::Real=1.0)
  y, v = prox_naive(g.f, x/gamma, 1.0/gamma)
  return x - gamma*y, vecdot(x,y) - gamma*vecdot(y,y) - v
end

is_prox_accurate(f::Conjugate) = is_prox_accurate(f.f)
