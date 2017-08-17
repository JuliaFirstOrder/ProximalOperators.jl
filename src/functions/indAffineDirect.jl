### CONCRETE TYPE: DIRECT PROX EVALUATION
# prox! is computed using a QR factorization of A'.

struct IndAffineDirect{R <: Real, T <: RealOrComplex{R}, M <: AbstractMatrix{T}, V <: AbstractVector{T}, F <: Factorization} <: IndAffine
  A::M
  b::V
  fact::F
  res::V
  function IndAffineDirect{R, T, M, V, F}(A::M, b::V) where {R <: Real, T <: RealOrComplex{R}, M <: AbstractMatrix{T}, V <: AbstractVector{T}, F <: Factorization}
    if size(A,1) > size(A,2)
      error("A must be full row rank")
    end
    normrowsinv = 1./vec(sqrt.(sum(abs2.(A), 2)))
    A = normrowsinv.*A # normalize rows of A
    b = normrowsinv.*b # and b accordingly
    fact = qrfact(A')
    new(A, b, fact, similar(b))
  end
end

is_cone(f::IndAffineDirect) = norm(f.b) == 0.0

IndAffineDirect(A::M, b::V) where {R <: Real, T <: RealOrComplex{R}, M <: DenseMatrix{T}, V <: AbstractVector{T}} = IndAffineDirect{R, T, M, V, LinAlg.QRCompactWY{T, M}}(A, b)

IndAffineDirect(A::M, b::V) where {R <: Real, T <: RealOrComplex{R}, I <: Integer, M <: SparseMatrixCSC{T, I}, V <: AbstractVector{T}} = IndAffineDirect{R, T, M, V, SparseArrays.SPQR.Factorization{T}}(A, b)

IndAffineDirect(a::V, b::T) where {R <: Real, T <: RealOrComplex{R}, V <: AbstractVector{T}} = IndAffineDirect(reshape(a,1,:), [b])

function (f::IndAffineDirect{R, T, M, V, F})(x::V) where {R, T, M, V, F}
  A_mul_B!(f.res, f.A, x)
  f.res .= f.b .- f.res
  # the tolerance in the following line should be customizable
  if norm(f.res, Inf) <= 1e-14
    return zero(R)
  end
  return typemax(R)
end

function prox!(y::V, f::IndAffineDirect{R, T, M, V, F}, x::V, gamma::R=one(R)) where {R, T, M, V, F <: LinAlg.QRCompactWY}
  A_mul_B!(f.res, f.A, x)
  f.res .= f.b .- f.res
  Rfact = view(f.fact.factors, 1:length(f.b), 1:length(f.b))
  LinAlg.LAPACK.trtrs!('U', 'C', 'N', Rfact, f.res)
  LinAlg.LAPACK.trtrs!('U', 'N', 'N', Rfact, f.res)
  Ac_mul_B!(y, f.A, f.res)
  y .+= x
  return zero(R)
end

# ### SPARSE CONCRETE TYPE
#
# struct IndAffineSparse{R <: Real, T <: RealOrComplex{R}, M <: AbstractSparseMatrix{T}, V <: AbstractVector{T}} <: IndAffine
#   A::M
#   b::V
#   F_Ac::SparseArrays.SPQR.Factorization{T}
#   res::V
#   function IndAffineSparse{R, T, M, V}(A::M, b::V) where {R <: Real, T <: RealOrComplex{R}, M <: AbstractSparseMatrix{T}, V <: AbstractVector{T}}
#     if size(A,1) > size(A,2)
#       error("A must be full row rank")
#     end
#     normrowsinv = 1./vec(sqrt.(sum(abs2.(A), 2)))
#     A = normrowsinv.*A # normalize rows of A
#     b = normrowsinv.*b # and b accordingly
#     F_Ac = qrfact(A') # factor AE = QR
#     new(A, b, F_Ac, similar(b))
#   end
# end
#
# is_cone(f::IndAffineSparse) = norm(f.b) == 0.0
#
# function (f::IndAffineSparse{R, T, M, V})(x::V) where {R, T, M, V}
#   A_mul_B!(f.res, f.A, x)
#   f.res .= f.b .- f.res
#   # the tolerance in the following line should be customizable
#   if norm(f.res, Inf) <= 1e-14
#     return zero(R)
#   end
#   return typemax(R)
# end

function prox!(y::V, f::IndAffineDirect{R, T, M, V, F}, x::V, gamma::R=one(R)) where {R, T, M, V, F <: SparseArrays.SPQR.Factorization}
  RTX_EQUALS_ETB = Int32(3)
  RETX_EQUALS_B  = Int32(1)
  spsolve = SparseArrays.SPQR.solve
  A_mul_B!(f.res, f.A, x)
  f.res .= f.b .- f.res
  RES = SparseArrays.CHOLMOD.Dense(f.res)
  # QR=AE so tmp = R'\E'res, RRres = E*R\(tmp), i.e. RRres = E*R\(R'\(E'res))
  RRres = convert(typeof(y), spsolve(RETX_EQUALS_B, f.fact, spsolve(RTX_EQUALS_ETB, f.fact, RES)))
  Ac_mul_B!(y, f.A, RRres)
  y .+= x
  return zero(R)
end

# # Not really naive, comparison is done to non-sparse in tests
# function prox_naive(f::IndAffineDirect{R, T, M, V, F}, x::V, gamma::R=one(R)) where {R, T, M, V, F <: SparseArrays.SPQR.Factorization{T}}
#   res = SparseArrays.CHOLMOD.Dense(f.A*x - f.b)
#   y = x - f.A'*convert(typeof(x), SparseArrays.SPQR.solve(Int32(1), f.F_Ac, SparseArrays.SPQR.solve(Int32(3), f.F_Ac, res)))
#   return y, zero(R)
# end

function prox_naive{R <: Real, T <: RealOrComplex{R}}(f::IndAffineDirect, x::AbstractArray{T,1}, gamma::R=one(R))
  y = x + f.A'*((f.A*f.A')\(f.b - f.A*x))
  return y, zero(R)
end
