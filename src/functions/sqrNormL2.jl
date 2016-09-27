# squared L2 norm (times a constant, or weighted)

immutable SqrNormL2{T <: Union{Real,AbstractArray}} <: ProximableFunction
  lambda::T
  SqrNormL2(lambda::T) =
    any(lambda .< 0) ? error("coefficients in λ must be nonnegative") : new(lambda)
end

"""
  SqrNormL2(λ::Real=1.0)

Returns the function `g(x) = (λ/2)(x'x)`, for a real parameter `λ ⩾ 0`.
"""

SqrNormL2() = SqrNormL2{Float64}(1.0)

SqrNormL2{T <: Real}(lambda::T) = SqrNormL2{T}(lambda)

"""
  SqrNormL2(λ::Array{Real})

Returns the function `g(x) = (1/2)(λ.*x)'x`, for an array of real parameters `λ ⩾ 0`.
"""

SqrNormL2{T <: Real}(lambda::AbstractArray{T}) = SqrNormL2{AbstractArray{T}}(lambda)

@compat function (f::SqrNormL2{S}){S <: Real, T <: RealOrComplex}(x::AbstractArray{T})
  return (f.lambda/2)*vecnorm(x)^2
end

@compat function (f::SqrNormL2{AbstractArray{S}}){S <: Real, T <: RealOrComplex}(x::AbstractArray{T})
  return 0.5*real(vecdot(f.lambda.*x,x))
end

function prox!{S <: Real, T <: Real}(f::SqrNormL2{S}, x::AbstractArray{T}, y::AbstractArray{T}, gamma::Real=1.0)
  gl = gamma*f.lambda
  sqny = zero(Float64)
  for k in eachindex(x)
    y[k] = x[k]/(1+gl)
    sqny += y[k]*y[k]
  end
  return (f.lambda/2)*sqny
end

function prox!{S <: Real, T <: Real}(f::SqrNormL2{AbstractArray{S}}, x::AbstractArray{T}, y::AbstractArray{T}, gamma::Real=1.0)
  wsqny = zero(Float64)
  for k in eachindex(x)
    y[k] = x[k]/(1+gamma*f.lambda[k])
    wsqny += f.lambda[k]*(y[k]*y[k])
  end
  return 0.5*wsqny
end

function prox!{S <: Real, T <: Complex}(f::SqrNormL2{S}, x::AbstractArray{T}, y::AbstractArray{T}, gamma::Real=1.0)
  gl = gamma*f.lambda
  sqny = zero(Float64)
  for k in eachindex(x)
    y[k] = x[k]/(1+gl)
    sqny += abs2(y[k])
  end
  return (f.lambda/2)*sqny
end

function prox!{S <: Real, T <: Complex}(f::SqrNormL2{AbstractArray{S}}, x::AbstractArray{T}, y::AbstractArray{T}, gamma::Real=1.0)
  wsqny = zero(Float64)
  for k in eachindex(x)
    y[k] = x[k]/(1+gamma*f.lambda[k])
    wsqny += f.lambda[k]*abs2(y[k])
  end
  return 0.5*wsqny
end

fun_name{T <: Real}(f::SqrNormL2{T}) = "squared Euclidean norm"
fun_name{T <: Real}(f::SqrNormL2{AbstractArray{T}}) = "weighted squared Euclidean norm"
fun_type(f::SqrNormL2) = "Array{Complex} → Real"
fun_expr{T <: Real}(f::SqrNormL2{T}) = "x ↦ (λ/2)||x||^2"
fun_expr{T <: Real}(f::SqrNormL2{AbstractArray{T}}) = "x ↦ (1/2)sum( λ_i (x_i)^2 )"
fun_params{T <: Real}(f::SqrNormL2{T}) = "λ = $(f.lambda)"
fun_params{T <: Real}(f::SqrNormL2{AbstractArray{T}}) = string("λ = ", typeof(f.lambda), " of size ", size(f.lambda))

function prox_naive{T <: RealOrComplex}(f::SqrNormL2, x::AbstractArray{T}, gamma::Real=1.0)
  y = x./(1+f.lambda*gamma)
  return y, 0.5*real(vecdot(f.lambda.*y,y))
end
