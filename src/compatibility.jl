########################################################################
# BEFORE 0.5

if VERSION < v"0.5-"

# 'similar' doesn't yield a Symmetric or Hermitian object in 0.4

function prox(f::IndPSD, x::Symmetric, gamma::Real=1.0)
  y = Symmetric(similar(x))
  fy = prox!(f, x, y, gamma)
  return y, fy
end

function prox(f::IndPSD, x::Hermitian, gamma::Real=1.0)
  y = Hermitian(similar(x))
  fy = prox!(f, x, y, gamma)
  return y, fy
end

end

########################################################################
# AFTER 0.5 (included)

if VERSION >= v"0.5-"

include("utilities/symmetricpacked.jl")

@compat function (f::IndPSD){T <: Float64}(x::AbstractVector{T})
  Z = dspev!('N', 'L', copy(x))
  for i in 1:length(Z)
    #Do we allow for some tolerance here?
    if Z[i] <= -1e-14
      return +Inf
    end
  end
  return 0.0
end

function prox!(f::IndPSD, x::AbstractVector{Float64}, y::AbstractVector{Float64}, gamma::Real=1.0)
  y[:] = x              # Copy x since dspev! corrupts input
  (W, Z) = dspevV!('L', y)
  W = max(W, 0)         # NonNeg eigenvalues
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

end
