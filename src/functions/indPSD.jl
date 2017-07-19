# indicator of a PSD

export IndPSD

"""
**Indicator of the set of positive semi-definite cone**

    IndPSD(;scaling=false)

Returns the indicator of the set
```math
C = \\{ X : X \\succeq 0 \\}.
```
The argument to the function can be either a `Symmetric` or `Hermitian` object,
or an object of type `AbstractVector{Float64}` holding a symmetric matrix in (lower triangular) packed storage.

If `scaling = true` then the vectors `y` and `x` in
`prox!(y::AbstractVector{Float64}, f::IndPSD, x::AbstractVector{Float64}, args...)`
have the off-diagonal elements multiplied with `√2` to preserve inner products,
see Vandenberghe 2010: http://www.seas.ucla.edu/~vandenbe/publications/coneprog.pdf .

I.e. when when `scaling=true`, let `X,Y` be matrices and

`x = (X_{1,1}, √2⋅X_{2,1}, ... ,√2⋅X_{n,1}, X_{2,2}, √2⋅X_{3,2}, ..., X_{n,n})`,

`y = (Y_{1,1}, √2⋅Y_{2,1}, ... ,√2⋅Y_{n,1}, Y_{2,2}, √2⋅Y_{3,2}, ..., Y_{n,n})`

then `prox!(Y, f, X)` is equivalent to `prox!(y, f, x)`.
"""

immutable IndPSD <: ProximableFunction
    scaling::Bool
end

IndPSD(;scaling=false) = IndPSD(scaling)

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

"""
Scales the diagonal of `x` with `val`, where `x` is the lower triangualar part
of a matrix, stored column by column.
"""
function scale_diagonal!(x, val)
    n = Int(sqrt(1/4+2*length(x))-1/2)
    k = -n
    for i = 1:n
         k += n - i + 2    #Calculate indices of diagonal elements recursively (paralell faster?)
         x[k] .*= val      #Scale diagonal
    end
    return
end

### Below: with AbstractVector argument

function (f::IndPSD){T <: Float64}(x::AbstractVector{T})
  y = copy(x)
  f.scaling && scale_diagonal!(y, sqrt(2)) #If scaling, scale diagonal (eigenvalues scaled by sqrt(2))

  Z = dspev!('N', 'L', y)
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

  f.scaling && scale_diagonal!(y, sqrt(2)) #If scaling, scale diagonal

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

  f.scaling && scale_diagonal!(y, 1/sqrt(2))  #If scaling, un-scale diagonal

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
  # Scale diagonal elements by sqrt(2)
  # See Vandenberghe 2010 http://www.seas.ucla.edu/~vandenbe/publications/coneprog.pdf
  # It's equivalent to scaling off-diagonal by 1/sqrt(2) and working with sqrt(2)*X
  if f.scaling
    for i = 1:n
      X[i,i] .*= sqrt(2)
    end
  end
  X, v = prox_naive(f, Symmetric(X), gamma)

  if f.scaling  #Scale diagonal elements back
    for i = 1:n
      X[i,i] ./= sqrt(2)
    end
  end

  y = similar(x)
  k = 1
  for j = 1:n, i = j:n                # Store Lower half of X in y
    y[k] = X[i,j]
    k = k+1
  end
  return y, 0.0
end
