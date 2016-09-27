# nuclear Norm (times a constant)

"""
  NuclearNorm(λ::Real=1.0)
  Returns the function `λ∑σ_i(X)`, where `σ_i(X)` is i-th singular
  value of matrix X.
"""

immutable NuclearNorm <: NormFunction
  lambda::Real
  NuclearNorm(lambda::Real=1.0) =
    lambda < 0 ? error("parameter λ must be nonnegative") : new(lambda)
end

@compat function (f::NuclearNorm){T <: RealOrComplex}(X::AbstractArray{T,2})
  U, S, V = svd(X);
  return f.lambda * sum(S);
end

function prox!{T <: RealOrComplex}(f::NuclearNorm, X::AbstractArray{T,2}, gamma::Real, Y::AbstractArray{T,2})
  U, S, V = svd(X)

  for i in eachindex(S)
    S[i] = max(0, S[i] - f.lambda*gamma);
  end

  Y[:] = U * diagm(S) * V'
  return f.lambda * sum(S);
end

fun_name(f::NuclearNorm) = "nuclear norm"
fun_type(f::NuclearNorm) = "Array{Complex,2} → Real"
fun_expr(f::NuclearNorm) = "X ↦ λ∑σ_i(X)"
fun_params(f::NuclearNorm) = "λ = $(f.lambda)"

function prox_naive{T <: RealOrComplex}(f::NuclearNorm, X::AbstractArray{T,2}, gamma::Real=1.0)
  U,S,V = svd(X)
  ftemp = NormL1(1.0)
  S_γ, fS_γ =  prox(ftemp,S,f.lambda*gamma)
  Y = U * diagm(S_γ) * V'
  return Y, f.lambda * sum(S_γ)
end
