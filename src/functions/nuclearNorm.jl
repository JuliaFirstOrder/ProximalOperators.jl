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

NuclearNorm{R <: Real}(lambda::R=1.0) = NuclearNorm{R}(lambda)

function (f::NuclearNorm){T <: RealOrComplex}(X::AbstractArray{T,2})
  U, S, V = svd(X);
  return f.lambda * sum(S);
end

function prox!{T <: RealOrComplex}(Y::AbstractArray{T,2}, f::NuclearNorm, X::AbstractArray{T,2}, gamma::Real=1.0)
  U, S, V = svd(X)

  for i in eachindex(S)
    S[i] = max(0, S[i] - f.lambda*gamma);
  end

  Y[:] = U * diagm(S) * V'
  return f.lambda * sum(S);
end

fun_name(f::NuclearNorm) = "nuclear norm"
fun_dom(f::NuclearNorm) = "AbstractArray{Real,2}, AbstractArray{Complex,2}"
fun_expr(f::NuclearNorm) = "X ↦ λ∑σ_i(X)"
fun_params(f::NuclearNorm) = "λ = $(f.lambda)"

function prox_naive{T <: RealOrComplex}(f::NuclearNorm, X::AbstractArray{T,2}, gamma::Real=1.0)
  U,S,V = svd(X)
  ftemp = NormL1(1.0)
  S_γ, fS_γ =  prox(ftemp,S,f.lambda*gamma)
  Y = U * diagm(S_γ) * V'
  return Y, f.lambda * sum(S_γ)
end
