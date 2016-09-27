# L2 norm (times a constant)

"""
  NormL2(λ::Real=1.0)

Returns the function `g(x) = λ||x||_2`, for a real parameter `λ ⩾ 0`.
"""

immutable NormL2 <: NormFunction
  lambda::Real
  NormL2(lambda::Real=1.0) =
    lambda < 0 ? error("parameter λ must be nonnegative") : new(lambda)
end

@compat function (f::NormL2){T <: RealOrComplex}(x::AbstractArray{T})
  return f.lambda*vecnorm(x)
end

function prox!{T <: RealOrComplex}(f::NormL2, x::AbstractArray{T}, y::AbstractArray{T}, gamma::Real=1.0)
  vecnormx = vecnorm(x)
  scale = max(0, 1-f.lambda*gamma/vecnormx)
  for i in eachindex(x)
    y[i] = scale*x[i]
  end
  return f.lambda*scale*vecnormx
end

fun_name(f::NormL2) = "Euclidean norm"
fun_type(f::NormL2) = "Array{Complex} → Real"
fun_expr(f::NormL2) = "x ↦ λ||x||_2"
fun_params(f::NormL2) = "λ = $(f.lambda)"

function prox_naive{T <: RealOrComplex}(f::NormL2, x::AbstractArray{T}, gamma::Real=1.0)
  vecnormx = vecnorm(x)
  scale = max(0, 1-f.lambda*gamma/vecnormx)
  y = scale*x
  return y, f.lambda*scale*vecnormx
end
