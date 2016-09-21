# L1 norm (times a constant, or weighted)

immutable NormL1{T <: Union{Real,RealArray}} <: NormFunction
  lambda::T
  NormL1(lambda) =
    any(lambda .< 0) ? error("coefficients in λ must be nonnegative") : new(lambda)
end

"""
  NormL1(λ::Real=1.0)

Returns the function `g(x) = λ||x||_1`, for a real parameter `λ ⩾ 0`.
"""

NormL1(lambda::Real=1.0) = NormL1{Real}(lambda)

"""
  NormL1(λ::Array{Real})

Returns the function `g(x) = sum(λ_i|x_i|, i = 1,...,n)`, for a vector of real
parameters `λ_i ⩾ 0`.
"""

NormL1(lambda::RealArray) = NormL1{RealArray}(lambda)

@compat function (f::NormL1{Real})(x::RealOrComplexArray)
  return f.lambda*vecnorm(x,1)
end

@compat function (f::NormL1{RealArray})(x::RealOrComplexArray)
  return vecnorm(f.lambda.*x,1)
end

function prox!(f::NormL1{RealArray}, x::RealArray, gamma::Real, y::RealArray)
  fy = zero(Float64)
  for i in eachindex(x)
    gl = gamma*f.lambda[i]
    y[i] = x[i] + (x[i] <= -gl ? gl : (x[i] >= gl ? -gl : -x[i]))
    fy += f.lambda[i]*abs(y[i])
  end
  return fy
end

function prox!(f::NormL1{RealArray}, x::ComplexArray, gamma::Real, y::ComplexArray)
  fy = zero(Float64)
  for i in eachindex(x)
    gl = gamma*f.lambda[i]
    y[i] = sign(x[i])*(abs(x[i]) <= gl ? 0 : abs(x[i]) - gl)
    fy += f.lambda[i]*abs(y[i])
  end
  return fy
end

function prox!(f::NormL1{Real}, x::RealArray, gamma::Real, y::RealArray)
  gl = gamma*f.lambda
  n1y = zero(Float64)
  for i in eachindex(x)
    y[i] = x[i] + (x[i] <= -gl ? gl : (x[i] >= gl ? -gl : -x[i]))
    n1y += abs(y[i])
  end
  return f.lambda*n1y
end

function prox!(f::NormL1{Real}, x::ComplexArray, gamma::Real, y::ComplexArray)
  gl = gamma*f.lambda
  n1y = zero(Float64)
  for i in eachindex(x)
    y[i] = sign(x[i])*(abs(x[i]) <= gl ? 0 : abs(x[i]) - gl)
    n1y += abs(y[i])
  end
  return f.lambda*n1y
end

fun_name(f::NormL1{Real}) = "L1 norm"
fun_name(f::NormL1{RealArray}) = "weighted L1 norm"
fun_type(f::NormL1{RealArray}) = "Array{Complex} → Real"
fun_expr(f::NormL1{Real}) = "x ↦ λ||x||_1"
fun_expr(f::NormL1{RealArray}) = "x ↦ sum( λ_i |x_i| )"
fun_params(f::NormL1{Real}) = "λ = $(f.lambda)"
fun_params(f::NormL1{RealArray}) = string("λ = ", typeof(f.lambda), " of size ", size(f.lambda))

function prox_naive(f::NormL1, x::RealOrComplexArray, gamma::Real=1.0)
  y = sign(x).*max(0.0, abs(x)-gamma*f.lambda)
  return y, vecnorm(f.lambda.*y,1)
end
