# L2 norm (times a constant)

"""
  NormL2(λ::Float64=1.0)

Returns the function `g(x) = λ||x||_2`, for a real parameter `λ ⩾ 0`.
"""

immutable NormL2 <: NormFunction
  lambda::Float64
  NormL2(lambda::Float64=1.0) =
    lambda < 0 ? error("parameter λ must be nonnegative") : new(lambda)
end

function call(f::NormL2, x::Array{Float64})
  return f.lambda*vecnorm(x)
end

function prox(f::NormL2, gamma::Float64, x::Array{Float64})
  vecnormx = vecnorm(x)
  scale = max(0, 1-f.lambda*gamma/vecnormx)
  y = scale*x
  return y, f.lambda*scale*vecnormx
end

fun_name(f::NormL2) = "weighted Euclidean norm"
fun_type(f::NormL2) = "R^n → R"
fun_expr(f::NormL2) = "x ↦ λ||x||_2"
fun_params(f::NormL2) = "λ = $(f.lambda)"
