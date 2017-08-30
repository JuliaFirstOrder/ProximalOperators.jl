# nuclear Norm (times a constant)

export NuclearNorm

"""
**Nuclear norm**

    NuclearNorm(λ=1.0)

Returns the function
```math
f(X) = \\|X\\|_* = λ ∑_i σ_i(X),
```
where `λ` is a positive parameter and ``σ_i(X)`` is ``i``-th singular value of matrix ``X``.
"""

immutable NuclearNorm{R <: Real} <: ProximableFunction
  lambda::R
  function NuclearNorm{R}(lambda::R) where {R <: Real}
    if lambda < 0
      error("parameter λ must be nonnegative")
    else
      new(lambda)
    end
  end
end

is_convex(f::NuclearNorm) = true

NuclearNorm(lambda::R=1.0) where {R <: Real} = NuclearNorm{R}(lambda)

function (f::NuclearNorm{R})(X::AbstractMatrix{T}) where {R <: Real, T <: Union{R, Complex{R}}}
  U, S, V = svd(X);
  return f.lambda * sum(S);
end

function prox!(Y::AbstractMatrix{T}, f::NuclearNorm{R}, X::AbstractMatrix{T}, gamma::R=one(R)) where {R <: Real, T <: Union{R, Complex{R}}}
  F = svdfact(X)
  S_thresh = max.(zero(R), F[:S] .- f.lambda*gamma)
  rankY = findfirst(S_thresh .== zero(R))
  if rankY == 0
    rankY = minimum(size(X))
  end
  Vt_thresh = view(F[:Vt], 1:rankY, :)
  U_thresh = view(F[:U], :, 1:rankY)
  M = S_thresh[1:rankY] .* Vt_thresh
  A_mul_B!(Y, U_thresh, M)
  return f.lambda * sum(S_thresh);
end

fun_name(f::NuclearNorm) = "nuclear norm"
fun_dom(f::NuclearNorm) = "AbstractArray{Real,2}, AbstractArray{Complex,2}"
fun_expr(f::NuclearNorm) = "X ↦ λ∑σ_i(X)"
fun_params(f::NuclearNorm) = "λ = $(f.lambda)"

function prox_naive(f::NuclearNorm, X::AbstractMatrix{T}, gamma=1.0) where T
  U, S, V = svd(X)
  S = max.(0, S .- f.lambda*gamma)
  Y = U * (spdiagm(S) * V')
  return Y, f.lambda * sum(S)
end
