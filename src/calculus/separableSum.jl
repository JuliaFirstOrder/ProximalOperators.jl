# Separable sum

immutable SeparableSum{T <: ElementwiseFunction} <: SeparableFunction
  g::T
end

@compat function (f::SeparableSum){T <: RealOrComplex}(x::AbstractArray{T})
  v = 0.0
  z = zeros(T, 1)
  for k in eachindex(x)
    z[1] = x[k]
    v += f.g(z)
  end
  return v
end

function prox!{T <: RealOrComplex, R <: Real}(f::SeparableSum, x::AbstractArray{T}, y::AbstractArray{T}, gamma::R=1.0)
  v = 0.0
  z = zeros(T, 1)
  for k in eachindex(x)
    z[1] = x[k]
    v += prox!(f.g, z, gamma)
    y[k] = z[1]
  end
  return v
end

function prox!{T <: RealOrComplex, R <: Real}(f::SeparableSum, x::AbstractArray{T}, y::AbstractArray{T}, gamma::AbstractArray{R})
  v = 0.0
  z = zeros(T, 1)
  for k in eachindex(x)
    z[1] = x[k]
    v += prox!(f.g, z, gamma[k])
    y[k] = z[1]
  end
  return v
end

is_prox_accurate(f::SeparableSum) = is_prox_accurate(f.g)
