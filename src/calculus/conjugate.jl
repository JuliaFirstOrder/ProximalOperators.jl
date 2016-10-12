# Conjugate

immutable Conjugate{T <: ProximableFunction} <: ProximableFunction
  f::T
end

# only prox! is provided here, call method would require being able to compute
# an element of the subdifferential of the conjugate

function prox!{T <: RealOrComplex}(g::Conjugate, x::AbstractArray{T}, y::AbstractArray{T}, gamma::Real=1.0)
  # Moreau identity
  v = prox!(g.f, x/gamma, y, 1.0/gamma)
  v = vecdot(x,y) - gamma*vecdot(y,y) - v
  y[:] *= -gamma
  y[:] += x
  return v
end
