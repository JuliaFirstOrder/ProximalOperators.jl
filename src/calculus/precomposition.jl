# Precomposition with scaling and translation

immutable Precomposition{T <: ProximableFunction, R <: Union{Real, AbstractArray}, S <: Union{Real, AbstractArray}} <: ProximableFunction
  f::T
  a::R
  b::S
  function Precomposition(f::T, a::R, b::S)
    if !(eltype(a) <: Real && eltype(b) <: Real)
      error("a and b must be real")
    end
    if any(a == 0.0)
      error("elements of a must be nonzero")
    else
      new(f, a, b)
    end
  end
end

Precomposition{T <: SeparableFunction, R <: AbstractArray, S <: AbstractArray}(f::T, a::R, b::S) = Precomposition{T, R, S}(f, a, b)

Precomposition{T <: SeparableFunction, R <: AbstractArray, S <: Real}(f::T, a::R, b::S=0.0) = Precomposition{T, R, S}(f, a, b)

Precomposition{T <: ProximableFunction, S <: Real}(f::T, a::S=1.0, b::S=0.0) = Precomposition{T, S, S}(f, a, b)

@compat function (g::Precomposition{T, S, V}){T <: ProximableFunction, S <: Real, V <: Union{Real, AbstractArray}, R <: RealOrComplex}(x::AbstractArray{R})
  return g.f((g.a)*x + g.b)
end

@compat function (g::Precomposition{T, S, V}){T <: SeparableFunction, S <: AbstractArray, V <: Union{Real, AbstractArray}, R <: RealOrComplex}(x::AbstractArray{R})
  return g.f((g.a).*x + g.b)
end

function prox!{T <: RealOrComplex, R <: Real}(g::Precomposition, x::AbstractArray{T}, y::AbstractArray{T}, gamma::R=1.0)
  y[:] = g.a .* x + g.b
  v = prox!(g.f, y, y, (g.a .* g.a) * gamma)
  y[:] -= g.b
  y[:] ./= g.a
  return v
end

function prox!{T <: RealOrComplex, R <: Real}(g::Precomposition, x::AbstractArray{T}, y::AbstractArray{T}, gamma::AbstractArray{R})
  y[:] = g.a .* x + g.b
  v = prox!(g.f, y, y, (g.a .* g.a) .* gamma)
  y[:] -= g.b
  y[:] ./= g.a
  return v
end
