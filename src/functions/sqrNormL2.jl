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

function prox(f::SqrNormL2{Float64}, x::RealOrComplexArray, gamma::Float64=1.0)
  y = x/(1+f.lambda*gamma)
  return y, (f.lambda/2)*vecnorm(y)^2
end

function prox(f::SqrNormL2, x::RealOrComplexArray, gamma::Float64=1.0)
  y = x./(1+f.lambda*gamma)
  return y, 0.5*real(vecdot(f.lambda.*y,y))
end

fun_name(f::SqrNormL2{Float64}) = "squared Euclidean norm"
fun_name(f::SqrNormL2) = "weighted squared Euclidean norm"
fun_type(f::SqrNormL2) = "C^n → R"
fun_expr(f::SqrNormL2{Float64}) = "x ↦ (λ/2)||x||^2"
fun_expr(f::SqrNormL2) = "x ↦ (1/2)sum( λ_i (x_i)^2 )"
fun_params(f::SqrNormL2{Float64}) = "λ = $(f.lambda)"
fun_params(f::SqrNormL2) = string("λ = ", typeof(f.lambda), " of size ", size(f.lambda))
