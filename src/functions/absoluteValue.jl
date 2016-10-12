# Absolute value

immutable AbsoluteValue <: ElementwiseFunction end

@compat function (f::AbsoluteValue){R <: Real}(x::AbstractArray{R})
  return x[1] > 0 ? x[1] : -x[1]
end

function prox!{R <: Real}(f::AbsoluteValue, x::AbstractArray{R}, y::AbstractArray{R}, gamma::Real)
  y[1] = x[1] > gamma ? x[1]-gamma : (x[1] < -gamma ? x[1]+gamma : 0.0)
  return y[1] > 0 ? y[1] : -y[1]
end
