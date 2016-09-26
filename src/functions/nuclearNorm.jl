# nuclear Norm (times a constant)

"""
  nuclearNorm(λ::Real=1.0)
  Returns the function `λ∑σ_i(X)`, where `σ_i(X)` is i-th singular
  value of matrix X.
"""

immutable nuclearNorm <: NormFunction
  lambda::Real
  nuclearNorm(lambda::Real=1.0) =
    lambda < 0 ? error("parameter λ must be nonnegative") : new(lambda)
end

@compat function (f::nuclearNorm){T <: RealOrComplex}(X::AbstractArray{T,2})
  U,S,V = svd(X);
  return f.lambda * sum(S);
end

function prox!{T <: RealOrComplex}(f::nuclearNorm, X::AbstractArray{T,2}, gamma::Real, Y::AbstractArray{T,2})
  U,S,V = svd(X)

  for i in eachindex(S)
    S[i] = max(0, S[i] - f.lambda*gamma);
  end

  Y[:] = U * diagm(S) * V'
  return f.lambda * sum(S);
end

fun_name(f::nuclearNorm) = "Nuclear Norm"
fun_type(f::nuclearNorm) = "Array{Complex,2} → Real"
fun_expr(f::nuclearNorm) = "X ↦ λ∑σ_i(X)"
fun_params(f::nuclearNorm) = "λ = $(f.lambda)"

function prox_naive{T <: RealOrComplex}(f::nuclearNorm, X::AbstractArray{T,2}, gamma::Real=1.0)
  U,S,V = svd(X)
  ftemp = NormL1(1.0)
  S_γ, fS_γ =  prox(ftemp,S,f.lambda*gamma)
  Y = U * diagm(S_γ) * V'
  return Y, f.lambda * sum(S_γ)
end
