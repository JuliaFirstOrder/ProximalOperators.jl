# indicator of a PSD

export IndPSD

"""
  IndPSD()

Returns the function `g = ind{A : A ⪰ 0}`, i.e. the indicator of the positive semidefinite cone.
The argument to the function can be either a Symmetric or Hermitian object.
From Julia 0.5, the argument can also be an AbstractVector{Float64} holding a symmetric matrix in (lower triangular) packed storage.
"""

immutable IndPSD <: ProximableFunction end

function (f::IndPSD){T <: RealOrComplex}(X::HermOrSym{T})
  F = eigfact(X);
  for i in eachindex(F.values)
    #Do we allow for some tolerance here?
    if F.values[i] <= -1e-14
      return +Inf
    end
  end
  return 0.0
end

is_convex(f::IndPSD) = true
is_cone(f::IndPSD) = true

function prox!{T <: RealOrComplex}(Y::HermOrSym{T}, f::IndPSD, X::HermOrSym{T}, gamma::Real=1.0)
  n = size(X, 1);
  F = eigfact(X);
  for i in eachindex(F.values)
    F.values[i] = max.(0.0, F.values[i]);
  end
  for i = 1:n
    for j = 1:n
      Y.data[i,j] = 0.0
      for k = 1:n
        Y.data[i,j] += F.vectors[i,k]*F.values[k]*F.vectors[j,k]
      end
    end
  end
  return 0.0
end

fun_name(f::IndPSD) = "indicator of positive semidefinite cone"
fun_dom(f::IndPSD) = "Symmetric, Hermitian, AbstractArray{Float64}"
fun_expr(f::IndPSD) = "x ↦ 0 if A ⪰ 0, +∞ otherwise"
fun_params(f::IndPSD) = "none"

function prox_naive{T <: RealOrComplex}(f::IndPSD, X::HermOrSym{T}, gamma::Real=1.0)
  F = eigfact(X);
  return F.vectors * diagm(max.(0.0, F.values)) * F.vectors', 0.0;
end

### Below: with AbstractVector argument

function (f::IndPSD){T <: Float64}(x::AbstractVector{T})
  Z = dspev!('N', 'L', copy(x))
  for i in 1:length(Z)
    #Do we allow for some tolerance here?
    if Z[i] <= -1e-14
      return +Inf
    end
  end
  return 0.0
end

function prox!(y::AbstractVector{Float64}, f::IndPSD, x::AbstractVector{Float64}, gamma::Real=1.0)
  y[:] = x              # Copy x since dspev! corrupts input
  (W, Z) = dspevV!('L', y)
  W = max.(W, 0.0)         # NonNeg eigenvalues
  M = Z.*W'             # Equivalent to Z*diagm(W) without constructing W matrix
  M = M*Z'              # Now let M = Z*diagm(W)*Z'
  n = length(W)
  k = 1
  for j in 1:n, i in j:n  # Store lower diagonal of M in y
    y[k] = M[i,j]
    k = k+1
  end
  return 0.0
end

function prox_naive{T<:Float64}(f::IndPSD, x::AbstractVector{T}, gamma::Real=1.0)
  n = Int(sqrt(1/4+2*length(x))-1/2)  # Formula for size of matrix
  X = Array{T,2}(n,n)
  k = 1
  for j = 1:n, i = j:n                # Store y in M
    X[i,j] = x[k]                   # Lower half
    if i != j
      X[j,i] = x[k]               # Strictly upper half
    end
    k = k+1
  end
  X, v = prox_naive(f, Symmetric(X), gamma)
  y = similar(x)
  k = 1
  for j = 1:n, i = j:n                # Store Lower half of X in y
    y[k] = X[i,j]
    k = k+1
  end
  return y, 0.0
end
