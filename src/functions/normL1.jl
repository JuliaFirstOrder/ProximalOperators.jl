# L1 norm (times a constant, or weighted)

immutable NormL1{T <: Union{Float64,Array{Float64}}} <: NormFunction
  lambda::T
  NormL1(lambda) =
    any(lambda .< 0) ? error("coefficients in λ must be nonnegative") : new(lambda)
end

"""
  NormL1(λ::Float64=1.0)

Returns the function `g(x) = λ||x||_1`, for a real parameter `λ ⩾ 0`.
"""

NormL1(lambda::Float64=1.0) = NormL1{Float64}(lambda)

"""
  NormL1(λ::Array{Float64})

Returns the function `g(x) = sum(λ_i|x_i|, i = 1,...,n)`, for a vector of real
parameters `λ_i ⩾ 0`.
"""

NormL1(lambda::Array{Float64}) = NormL1{Array{Float64}}(lambda)

@compat function (f::NormL1{Float64})(x::RealOrComplexArray)
  return f.lambda*vecnorm(x,1)
end

@compat function (f::NormL1)(x::RealOrComplexArray)
  return vecnorm(f.lambda.*x,1)
end

function prox(f::NormL1{Float64}, x::RealOrComplexArray, gamma::Float64=1.0)
  y = sign(x).*max(0.0, abs(x)-gamma*f.lambda)
  return y, f.lambda*vecnorm(y,1)
end

function prox(f::NormL1, x::RealOrComplexArray, gamma::Float64=1.0)
  y = sign(x).*max(0.0, abs(x)-gamma*f.lambda)
  return y, vecnorm(f.lambda.*y,1)
end

fun_name(f::NormL1{Float64}) = "L1 norm"
fun_name(f::NormL1) = "weighted L1 norm"
fun_type(f::NormL1) = "C^n → R"
fun_expr(f::NormL1{Float64}) = "x ↦ λ||x||_1"
fun_expr(f::NormL1) = "x ↦ sum( λ_i |x_i| )"
fun_params(f::NormL1{Float64}) = "λ = $(f.lambda)"
fun_params(f::NormL1) = string("λ = ", typeof(f.lambda), " of size ", size(f.lambda))
