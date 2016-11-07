# Hinge loss function

"""
  HingeLoss(b::Array{Real}, mu::Real=1.0)

Returns the function `g(x) = mu * sum(max(0, 1 - b_i * x_i), i=1,...,n )`.
"""

immutable HingeLoss{T <: AbstractArray, R <: Real} <: ProximableFunction
  b::T
  mu::R
  function HingeLoss(b::T, mu::R)
    if mu <= 0.0
      error("parameter mu must be positive")
    else
      new(b, mu)
    end
  end
end

HingeLoss{T <: AbstractArray, R <: Real}(b::T, mu::R=1.0) = HingeLoss{T, R}(b, mu)

@compat function (f::HingeLoss){T <: Real}(x::AbstractArray{T})
  return (f.mu)*sum(max(0.0, 1-(f.b).*x))
end

function prox!{T <: Real}(f::HingeLoss, x::AbstractArray{T}, y::AbstractArray{T}, gamma::Real=1.0)
  bx = 0.0
  fsum = 0.0
  for i in eachindex(x)
    bx = f.b[i]*x[i]
    y[i] = bx < 1 ? f.b[i]*min(bx+gamma*(f.mu), 1.0) : x[i]
    fsum += max(0.0, 1.0-f.b[i]*y[i])
  end
  return (f.mu)*fsum
end

fun_name(f::HingeLoss) = "hinge loss"
fun_dom(f::HingeLoss) = "AbstractArray{Real}"
fun_expr(f::HingeLoss) = "x ↦ μ * sum( max(0, 1 - b_i*x_i), i=1,...,n )"
fun_params(f::HingeLoss) = string("b = ", typeof(f.b), " of size ", size(f.b), ", μ = $(f.mu)")

function prox_naive{T <: Real}(f::HingeLoss, x::AbstractArray{T}, gamma::Real=1.0)
  y = similar(x)
  bx = (f.b).*x
  ind = bx .< 1
  y[ind] = f.b[ind].*min(bx[ind]+gamma*(f.mu), 1.0)
  y[!ind] = x[!ind]
  return y, (f.mu)*sum(max(0.0, 1.0-(f.b).*y));
end
