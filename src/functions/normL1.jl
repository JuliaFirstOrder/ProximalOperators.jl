# L1 norm (times a constant, or weighted)

immutable NormL1{T <: Union{Real, AbstractArray}} <: ProximableFunction
  lambda::T
  function NormL1(lambda::T)
    if !(eltype(lambda) <: Real)
      error("λ must be real")
    end
    if any(lambda .< 0)
      error("λ must be nonnegative")
    else
      new(lambda)
    end
  end
end

is_separable(f::NormL1) = true
is_convex(f::NormL1) = true

"""
  NormL1(λ::Real=1.0)

Returns the function `g(x) = λ||x||_1`, for a real parameter `λ ⩾ 0`.
"""

NormL1{R <: Real}(lambda::R=1.0) = NormL1{R}(lambda)

"""
  NormL1(λ::Array{Real})

Returns the function `g(x) = sum(λ_i|x_i|, i = 1,...,n)`, for a vector of real
parameters `λ_i ⩾ 0`.
"""

NormL1{A <: AbstractArray}(lambda::A) = NormL1{A}(lambda)

function (f::NormL1{R}){R <: Real}(x::AbstractArray)
  return f.lambda*vecnorm(x,1)
end

function (f::NormL1{A}){A <: AbstractArray}(x::AbstractArray)
  return vecnorm(f.lambda.*x,1)
end

function prox!{A <: AbstractArray, R <: Real}(y::AbstractArray{R}, f::NormL1{A}, x::AbstractArray{R}, gamma::Real=1.0)
  fy = zero(R)
  for i in eachindex(x)
    gl = gamma*f.lambda[i]
    y[i] = x[i] + (x[i] <= -gl ? gl : (x[i] >= gl ? -gl : -x[i]))
  end
  return sum(f.lambda .* abs.(y))
end

function prox!{A <: AbstractArray, R <: Real}(y::AbstractArray{Complex{R}}, f::NormL1{A}, x::AbstractArray{Complex{R}}, gamma::Real=1.0)
  fy = zero(R)
  for i in eachindex(x)
    gl = gamma*f.lambda[i]
    y[i] = sign(x[i])*(abs(x[i]) <= gl ? 0 : abs(x[i]) - gl)
  end
  return sum(f.lambda .* abs.(y))
end

function prox!{T <: Real, R <: Real}(y::AbstractArray{R}, f::NormL1{T}, x::AbstractArray{R}, gamma::Real=1.0)
  n1y = zero(R)
  gl = gamma*f.lambda
  for i in eachindex(x)
    y[i] = x[i] + (x[i] <= -gl ? gl : (x[i] >= gl ? -gl : -x[i]))
    n1y += y[i] > 0 ? y[i] : -y[i]
  end
  return f.lambda*n1y
end

function prox!{T <: Real, R <: Real}(y::AbstractArray{Complex{R}}, f::NormL1{T}, x::AbstractArray{Complex{R}}, gamma::Real=1.0)
  gl = gamma*f.lambda
  n1y = zero(R)
  for i in eachindex(x)
    y[i] = sign(x[i])*(abs(x[i]) <= gl ? 0 : abs(x[i]) - gl)
    n1y += abs(y[i])
  end
  return f.lambda*n1y
end

function prox!{A <: AbstractArray, R <: Real}(y::AbstractArray{R}, f::NormL1{A}, x::AbstractArray{R}, gamma::AbstractArray)
  fy = zero(R)
  for i in eachindex(x)
    gl = gamma[i]*f.lambda[i]
    y[i] = x[i] + (x[i] <= -gl ? gl : (x[i] >= gl ? -gl : -x[i]))
  end
  return sum(f.lambda .* abs.(y))
end

function prox!{A <: AbstractArray, R <: Real}(y::AbstractArray{Complex{R}}, f::NormL1{A}, x::AbstractArray{Complex{R}}, gamma::AbstractArray)
  fy = zero(R)
  for i in eachindex(x)
    gl = gamma[i]*f.lambda[i]
    y[i] = sign(x[i])*(abs(x[i]) <= gl ? 0 : abs(x[i]) - gl)
  end
  return sum(f.lambda .* abs.(y))
end

function prox!{T <: Real, R <: Real}(y::AbstractArray{R}, f::NormL1{T}, x::AbstractArray{R}, gamma::AbstractArray)
  n1y = zero(R)
  for i in eachindex(x)
    gl = gamma[i]*f.lambda
    y[i] = x[i] + (x[i] <= -gl ? gl : (x[i] >= gl ? -gl : -x[i]))
    n1y += y[i] > 0 ? y[i] : -y[i]
  end
  return f.lambda*n1y
end

function prox!{T <: Real, R <: Real}(y::AbstractArray{Complex{R}}, f::NormL1{T}, x::AbstractArray{Complex{R}}, gamma::AbstractArray)
  n1y = zero(R)
  for i in eachindex(x)
    gl = gamma[i]*f.lambda
    y[i] = sign(x[i])*(abs(x[i]) <= gl ? 0 : abs(x[i]) - gl)
    n1y += abs(y[i])
  end
  return f.lambda*n1y
end

fun_name(f::NormL1) = "weighted L1 norm"
fun_dom(f::NormL1) = "AbstractArray{Real}, AbstractArray{Complex}"
fun_expr{R <: Real}(f::NormL1{R}) = "x ↦ λ||x||_1"
fun_expr{A <: AbstractArray}(f::NormL1{A}) = "x ↦ sum( λ_i |x_i| )"
fun_params{R <: Real}(f::NormL1{R}) = "λ = $(f.lambda)"
fun_params{A <: AbstractArray}(f::NormL1{A}) = string("λ = ", typeof(f.lambda), " of size ", size(f.lambda))

function prox_naive{T <: RealOrComplex}(f::NormL1, x::AbstractArray{T}, gamma::Union{Real, AbstractArray}=1.0)
  y = sign.(x).*max.(0.0, abs.(x)-gamma.*f.lambda)
  return y, vecnorm(f.lambda.*y,1)
end
