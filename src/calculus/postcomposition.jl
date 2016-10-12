# Postcomposition

immutable Postcomposition{T <: ProximableFunction, S <: Real} <: ProximableFunction
  f::T
  a::S
  b::S
  function Postcomposition(f::T, a::S, b::S)
    if a <= 0.0
      error("parameter a must be positive")
    else
      new(f, a, b)
    end
  end
end

Postcomposition{T <: ProximableFunction, S <: Real}(f::T, a::S=1.0, b::S=0.0) = Postcomposition{T, S}(f, a, b)

@compat function (g::Postcomposition){T <: RealOrComplex}(x::AbstractArray{T})
  return g.a * g.f(x) + g.b
end

function prox!{T <: RealOrComplex}(g::Postcomposition, x::AbstractArray{T}, y::AbstractArray{T}, gamma::Real=1.0)
  v = prox!(g.f, x, y, g.a * gamma)
  return g.a * v + g.b
end
