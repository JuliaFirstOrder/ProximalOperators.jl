# Hinge loss function

export SqrHingeLoss

"""
**Squared Hinge loss**

  SqrHingeLoss(b, μ=1.0)

Returns the function
```math
f(x) = μ⋅∑_i \\max\\{0, 1 - b_i ⋅ x_i\\}^2,
```
where `b` is an array and `μ` is a positive parameter.
"""

struct SqrHingeLoss{R <: Real, T <:AbstractVector{R}} <: ProximableFunction
  b::T
  mu::R
  function SqrHingeLoss{R,T}(b::T, mu::R) where {R <: Real, T <:AbstractVector{R}}
    if mu <= 0
      error("parameter mu must be positive")
    else
      new(b, mu)
    end
  end
end

is_convex(f::SqrHingeLoss) = true
is_smooth(f::SqrHingeLoss) = true

SqrHingeLoss(b::T, mu::R=1.0) where {R <: Real, T <: AbstractVector{R}} = SqrHingeLoss{R,T}(b,mu)

# TODO prox!

(f::SqrHingeLoss){T <: Real}(x::AbstractArray{T}) = f.mu*sum( max.(zero(T),(1.-f.b.*x)).^2 )

function gradient!(y::AbstractArray{T}, f::SqrHingeLoss{T}, x::AbstractArray{T}) where {T <: Real}
	sum = zero(T)
	for i in eachindex(x)
		zz = 1-f.b[i]*x[i]
		z = max(zero(T),zz)
		y[i] = z .> 0 ? -2*f.mu*f.b[i]*zz : 0
		sum += z^2
	end
	return f.mu*sum
end

fun_name(f::SqrHingeLoss) = "squared hinge loss"
fun_dom(f::SqrHingeLoss) = "AbstractArray{Real}"
fun_expr(f::SqrHingeLoss) = "x ↦ μ * sum( max(0,1-b*x_i)^2, i=1,...,n )"
fun_params(f::SqrHingeLoss) = "b = $(typeof(f.b)), μ = $(f.mu)"
