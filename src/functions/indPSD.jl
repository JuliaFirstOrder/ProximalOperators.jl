# indicator of a PSD

"""
  IndPDS(a::HermOrSym{T})

Returns the function `g = ind{A : A ⪰ 0}`, i.e. an indicator of
positive semidefinite cone
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

function prox!{T <: RealOrComplex}(f::IndPSD, X::HermOrSym{T}, gamma::Real, Y::HermOrSym{T})
  F = eigfact(X);
  for i in eachindex(F.values)
    F.values[i] = max(0,F.values[i]);
  end
  Y.data[:] = F.vectors * diagm(F.values) * F.vectors'
  return 0.0
end

fun_name(f::IndPSD) = "indicator of Positive semidefinite cone"
fun_type(f::IndPSD) = "HermOrSym → Real ∪ {+∞}"
fun_expr(f::IndPSD) = "x ↦ 0 if A ⪰ 0,  +∞ otherwise"
fun_params(f::IndPSD) = "none"

function prox_naive{T <: RealOrComplex}(f::IndPSD, X::HermOrSym{T}, gamma::Real=1.0)
  F = eigfact(X);
  for i in eachindex(F.values)
    F.values[i] = max(0,F.values[i]);
  end
  return  F.vectors * diagm(F.values) * F.vectors', 0.0;
end
