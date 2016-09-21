# squared L2 norm (times a constant, or weighted)

immutable SqrNormL2{T <: Union{Real,RealArray}} <: ProximableFunction
  lambda::T
  SqrNormL2(lambda) =
    any(lambda .< 0) ? error("coefficients in λ must be nonnegative") : new(lambda)
end

"""
  SqrNormL2(λ::Real=1.0)

Returns the function `g(x) = (λ/2)(x'x)`, for a real parameter `λ ⩾ 0`.
"""

SqrNormL2(lambda::Real=1.0) = SqrNormL2{Real}(lambda)

"""
  SqrNormL2(λ::Array{Real})

Returns the function `g(x) = (1/2)(λ.*x)'x`, for an array of real parameters `λ ⩾ 0`.
"""

SqrNormL2(lambda::RealArray) = SqrNormL2{RealArray}(lambda)

@compat function (f::SqrNormL2{Real})(x::RealOrComplexArray)
  return (f.lambda/2)*vecnorm(x)^2
end

@compat function (f::SqrNormL2{RealArray})(x::RealOrComplexArray)
  return 0.5*real(vecdot(f.lambda.*x,x))
end

function prox!(f::SqrNormL2{Real}, x::RealArray, gamma::Real, y::RealArray)
  gl = gamma*f.lambda
  sqny = zero(Float64)
  for k in eachindex(x)
    y[k] = x[k]/(1+gl)
    sqny += y[k]*y[k]
  end
  return (f.lambda/2)*sqny
end

function prox!(f::SqrNormL2{RealArray}, x::RealArray, gamma::Real, y::RealArray)
  wsqny = zero(Float64)
  for k in eachindex(x)
    y[k] = x[k]/(1+gamma*f.lambda[k])
    wsqny += f.lambda[k]*(y[k]*y[k])
  end
  return 0.5*wsqny
end

function prox!(f::SqrNormL2{Real}, x::ComplexArray, gamma::Real, y::ComplexArray)
  gl = gamma*f.lambda
  sqny = zero(Float64)
  for k in eachindex(x)
    y[k] = x[k]/(1+gl)
    sqny += abs2(y[k])
  end
  return (f.lambda/2)*sqny
end

function prox!(f::SqrNormL2{RealArray}, x::ComplexArray, gamma::Real, y::ComplexArray)
  wsqny = zero(Float64)
  for k in eachindex(x)
    y[k] = x[k]/(1+gamma*f.lambda[k])
    wsqny += f.lambda[k]*abs2(y[k])
  end
  return 0.5*wsqny
end

fun_name(f::SqrNormL2{Real}) = "squared Euclidean norm"
fun_name(f::SqrNormL2{RealArray}) = "weighted squared Euclidean norm"
fun_type(f::SqrNormL2{RealArray}) = "Array{Complex} → Real"
fun_expr(f::SqrNormL2{Real}) = "x ↦ (λ/2)||x||^2"
fun_expr(f::SqrNormL2{RealArray}) = "x ↦ (1/2)sum( λ_i (x_i)^2 )"
fun_params(f::SqrNormL2{Real}) = "λ = $(f.lambda)"
fun_params(f::SqrNormL2{RealArray}) = string("λ = ", typeof(f.lambda), " of size ", size(f.lambda))

function prox_naive(f::SqrNormL2, x::RealOrComplexArray, gamma::Real=1.0)
  y = x./(1+f.lambda*gamma)
  return y, 0.5*real(vecdot(f.lambda.*y,y))
end
