# squared L2 norm (times a constant, or weighted)

immutable SqrNormL2{T <: Union{Float64,Array{Float64}}} <: ProximableFunction
  lambda::T
  SqrNormL2(lambda) =
    any(lambda .< 0) ? error("coefficients in λ must be nonnegative") : new(lambda)
end

"""
  SqrNormL2(λ::Float64=1.0)

Returns the function `g(x) = (λ/2)(x'x)`, for a real parameter `λ ⩾ 0`.
"""

SqrNormL2(lambda::Float64=1.0) = SqrNormL2{Float64}(lambda)

"""
  SqrNormL2(λ::Array{Float64})

Returns the function `g(x) = (1/2)(λ.*x)'x`, for an array of real parameters `λ ⩾ 0`.
"""

SqrNormL2(lambda::Array{Float64}) = SqrNormL2{Array{Float64}}(lambda)

@compat function (f::SqrNormL2{Float64})(x::RealOrComplexArray)
  return (f.lambda/2)*vecnorm(x)^2
end

@compat function (f::SqrNormL2)(x::RealOrComplexArray)
  return 0.5*real(vecdot(f.lambda.*x,x))
end

function prox!(f::SqrNormL2{Float64}, x::RealArray, gamma::Float64, y::RealArray)
  gl = gamma*f.lambda
  sqny = zero(Float64)
  for k in eachindex(x)
    y[k] = x[k]/(1+gl)
    sqny += y[k]*y[k]
  end
  return (f.lambda/2)*sqny
end

function prox!(f::SqrNormL2, x::RealArray, gamma::Float64, y::RealArray)
  wsqny = zero(Float64)
  for k in eachindex(x)
    y[k] = x[k]/(1+gamma*f.lambda[k])
    wsqny += f.lambda[k]*(y[k]*y[k])
  end
  return 0.5*wsqny
end

function prox!(f::SqrNormL2{Float64}, x::ComplexArray, gamma::Float64, y::ComplexArray)
  gl = gamma*f.lambda
  sqny = zero(Float64)
  for k in eachindex(x)
    y[k] = x[k]/(1+gl)
    sqny += abs2(y[k])
  end
  return (f.lambda/2)*sqny
end

function prox!(f::SqrNormL2, x::ComplexArray, gamma::Float64, y::ComplexArray)
  wsqny = zero(Float64)
  for k in eachindex(x)
    y[k] = x[k]/(1+f.lambda[k])
    wsqny += f.lambda[k]*abs2(y[k])
  end
  return 0.5*wsqny
end

fun_name(f::SqrNormL2{Float64}) = "squared Euclidean norm"
fun_name(f::SqrNormL2) = "weighted squared Euclidean norm"
fun_type(f::SqrNormL2) = "Array{Complex} → Real"
fun_expr(f::SqrNormL2{Float64}) = "x ↦ (λ/2)||x||^2"
fun_expr(f::SqrNormL2) = "x ↦ (1/2)sum( λ_i (x_i)^2 )"
fun_params(f::SqrNormL2{Float64}) = "λ = $(f.lambda)"
fun_params(f::SqrNormL2) = string("λ = ", typeof(f.lambda), " of size ", size(f.lambda))

function prox_naive(f::SqrNormL2, x::RealOrComplexArray, gamma::Float64=1.0)
  y = x./(1+f.lambda*gamma)
  return y, 0.5*real(vecdot(f.lambda.*y,y))
end
