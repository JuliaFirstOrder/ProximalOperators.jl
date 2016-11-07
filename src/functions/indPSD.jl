# indicator of a PSD

"""
  IndPSD()

Returns the function `g = ind{A : A ⪰ 0}`, i.e. the indicator of the positive semidefinite cone.
The argument to the function can be either a Symmetric or Hermitian object.
From Julia 0.5, the argument can also be an AbstractVector{Float64} holding a symmetric matrix in (lower triangular) packed storage.
"""

immutable IndPSD <: IndicatorConvex end

@compat function (f::IndPSD){T <: RealOrComplex}(X::HermOrSym{T})
  F = eigfact(X);
  for i in eachindex(F.values)
    #Do we allow for some tolerance here?
    if F.values[i] <= -1e-14
      return +Inf
    end
  end
  return 0.0
end

function prox!{T <: RealOrComplex}(f::IndPSD, X::HermOrSym{T}, Y::HermOrSym{T}, gamma::Real=1.0)
  F = eigfact(X);
  for i in eachindex(F.values)
    F.values[i] = max(0,F.values[i]);
  end
  Y.data[:] = F.vectors * diagm(F.values) * F.vectors'
  return 0.0
end

fun_name(f::IndPSD) = "indicator of positive semidefinite cone"
fun_dom(f::IndPSD) = "Symmetric, Hermitian, AbstractArray{Float64}"
fun_expr(f::IndPSD) = "x ↦ 0 if A ⪰ 0, +∞ otherwise"
fun_params(f::IndPSD) = "none"

function prox_naive{T <: RealOrComplex}(f::IndPSD, X::HermOrSym{T}, gamma::Real=1.0)
  F = eigfact(X);
  return F.vectors * diagm(max(0.0, F.values)) * F.vectors', 0.0;
end
