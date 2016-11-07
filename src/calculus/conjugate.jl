# Conjugate

immutable Conjugate{T <: ProximableFunction} <: ProximableFunction
  f::T
end

fun_dom(f::Conjugate) = fun_dom(f.f)

# only prox! is provided here, call method would require being able to compute
# an element of the subdifferential of the conjugate

function prox!{R <: Real}(g::Conjugate, x::AbstractArray{R}, y::AbstractArray{R}, gamma::Real=1.0)
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

# complex case, need to cast inner products to real

function prox!{R <: Real}(g::Conjugate, x::AbstractArray{Complex{R}}, y::AbstractArray{Complex{R}}, gamma::Real=1.0)
  x_copy = copy(x)
  v = prox!(g.f, x/gamma, y, 1.0/gamma)
  v = real(vecdot(x_copy,y)) - gamma*real(vecdot(y,y)) - v
  y[:] *= -gamma
  y[:] += x_copy
  return v
end

# special case for indicator of convex cones (the conjugate is also an indicator function)

prox!{R <: Real, C <: IndicatorConvexCone}(g::Conjugate{C}, x::AbstractArray{R}, y::AbstractArray{R}, gamma::Real=1.0) =
  prox_conjugate_convex_cone!(g.f, x, y, gamma)

prox!{R <: Real, C <: IndicatorConvexCone}(g::Conjugate{C}, x::AbstractArray{Complex{R}}, y::AbstractArray{Complex{R}}, gamma::Real=1.0) =
  prox_conjugate_convex_cone!(g.f, x, y, gamma)

function prox_conjugate_convex_cone!{T <: RealOrComplex, C <: IndicatorConvexCone}(f::C, x::AbstractArray{T}, y::AbstractArray{T}, gamma::Real=1.0)
  x_copy = copy(x)
  prox!(f, x/gamma, y, 1.0/gamma)
  y[:] *= -gamma
  y[:] += x_copy
  return 0.0
end

# naive implementation

function prox_naive{T <: RealOrComplex}(g::Conjugate, x::AbstractArray{T}, gamma::Real=1.0)
  y, v = prox_naive(g.f, x/gamma, 1.0/gamma)
  return x - gamma*y, real(vecdot(x,y)) - gamma*real(vecdot(y,y)) - v
end

is_prox_accurate(f::Conjugate) = is_prox_accurate(f.f)
