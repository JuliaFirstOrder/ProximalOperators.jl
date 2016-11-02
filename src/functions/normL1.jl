# L1 norm (times a constant, or weighted)

immutable NormL1{T <: Union{Real, AbstractArray}} <: NormFunction
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

"""
  NormL1(λ::Real=1.0)

Returns the function `g(x) = λ||x||_1`, for a real parameter `λ ⩾ 0`.
"""

NormL1{T <: Real}(lambda::T=1.0) = NormL1{T}(lambda)

"""
  NormL1(λ::Array{Real})

Returns the function `g(x) = sum(λ_i|x_i|, i = 1,...,n)`, for a vector of real
parameters `λ_i ⩾ 0`.
"""

NormL1{T <: AbstractArray}(lambda::T) = NormL1{T}(lambda)

@compat function (f::NormL1{S}){S <: Real, T <: RealOrComplex}(x::AbstractArray{T})
  return f.lambda*vecnorm(x,1)
end

@compat function (f::NormL1{S}){S <: AbstractArray, T <: RealOrComplex}(x::AbstractArray{T})
  return vecnorm(f.lambda.*x,1)
end

function prox!{S <: AbstractArray, T <: Real}(f::NormL1{S}, x::AbstractArray{T}, y::AbstractArray{T}, gamma::Real=1.0)
  fy = zero(Real)
  for i in eachindex(x)
    gl = gamma*f.lambda[i]
    y[i] = x[i] + (x[i] <= -gl ? gl : (x[i] >= gl ? -gl : -x[i]))
  end
  return sum(f.lambda .* abs(y))
end

function prox!{S <: AbstractArray, T <: Complex}(f::NormL1{S}, x::AbstractArray{T}, y::AbstractArray{T}, gamma::Real=1.0)
  fy = zero(Real)
  for i in eachindex(x)
    gl = gamma*f.lambda[i]
    y[i] = sign(x[i])*(abs(x[i]) <= gl ? 0 : abs(x[i]) - gl)
  end
  return sum(f.lambda .* abs(y))
end

function prox!{S <: Real, T <: Real}(f::NormL1{S}, x::AbstractArray{T}, y::AbstractArray{T}, gamma::Real=1.0)
  n1y = zero(Real)
  gl = gamma*f.lambda
  for i in eachindex(x)
    y[i] = x[i] + (x[i] <= -gl ? gl : (x[i] >= gl ? -gl : -x[i]))
    n1y += y[i] > 0 ? y[i] : -y[i]
  end
  return f.lambda*n1y
end

function prox!{S <: Real, T <: Complex}(f::NormL1{S}, x::AbstractArray{T}, y::AbstractArray{T}, gamma::Real=1.0)
  gl = gamma*f.lambda
  n1y = zero(Real)
  for i in eachindex(x)
    y[i] = sign(x[i])*(abs(x[i]) <= gl ? 0 : abs(x[i]) - gl)
    n1y += abs(y[i])
  end
  return f.lambda*n1y
end

fun_name(f::NormL1) = "weighted L1 norm"
fun_type(f::NormL1) = "Array{Complex} → Real"
fun_expr{T <: Real}(f::NormL1{T}) = "x ↦ λ||x||_1"
fun_expr{T <: AbstractArray}(f::NormL1{T}) = "x ↦ sum( λ_i |x_i| )"
fun_params{T <: Real}(f::NormL1{T}) = "λ = $(f.lambda)"
fun_params{T <: AbstractArray}(f::NormL1{T}) = string("λ = ", typeof(f.lambda), " of size ", size(f.lambda))

function prox_naive{T <: RealOrComplex}(f::NormL1, x::AbstractArray{T}, gamma::Real=1.0)
  y = sign(x).*max(0.0, abs(x)-gamma*f.lambda)
  return y, vecnorm(f.lambda.*y,1)
end
