### CONCRETE TYPE: DIRECT PROX EVALUATION

mutable struct QuadraticDirect{R <: Real, M <: AbstractMatrix{R}, V <: AbstractVector{R}, F <: Factorization} <: Quadratic
  Q::M
  q::V
  gamma::R
  temp::V
  fact::F
  function QuadraticDirect{R, M, V, F}(Q::M, q::V) where {R <: Real, M <: AbstractMatrix{R}, V <: AbstractVector{R}, F <: Factorization}
    if size(Q, 1) != size(Q, 2) || length(q) != size(Q, 2)
      error("Q must be squared and q must be compatible with Q")
    end
    new(Q, q, -1, similar(q))
  end
end

function QuadraticDirect(Q::M, q::V) where {R <: Real, I <: Integer, M <: SparseMatrixCSC{R, I}, V <: AbstractVector{R}}
  QuadraticDirect{R, M, V, SparseArrays.CHOLMOD.Factor{R}}(Q, q)
end

function QuadraticDirect(Q::M, q::V) where {R <: Real, M <: DenseMatrix{R}, V <: AbstractVector{R}}
  QuadraticDirect{R, M, V, LinAlg.Cholesky{R}}(Q, q)
end

function (f::QuadraticDirect{R, M, V, F})(x::AbstractArray{R}) where {R, M, V, F}
  A_mul_B!(f.temp, f.Q, x)
  return 0.5*vecdot(x, f.temp) + vecdot(x, f.q)
end

function prox!(y::AbstractArray{R}, f::QuadraticDirect{R, M, V, F}, x::AbstractArray{R}, gamma::R=one(R)) where {R, M, V, F <: LinAlg.Cholesky}
  if gamma != f.gamma
    factor_step!(f, gamma)
  end
  y .= x./gamma
  y .-= f.q
  # Qy = LL'y = b, therefore y = L'\(L\b)
  LinAlg.LAPACK.trtrs!('L', 'N', 'N', f.fact.factors, y)
  LinAlg.LAPACK.trtrs!('L', 'C', 'N', f.fact.factors, y)
  A_mul_B!(f.temp, f.Q, y)
  fy = 0.5*vecdot(y, f.temp) + vecdot(y, f.q)
  return fy
end

function prox!(y::AbstractArray{R}, f::QuadraticDirect{R, M, V, F}, x::AbstractArray{R}, gamma::R=one(R)) where {R, M, V, F <: SparseArrays.CHOLMOD.Factor}
  if gamma != f.gamma
    factor_step!(f, gamma)
  end
  f.temp .= x./gamma
  f.temp .-= f.q
  y .= f.fact\f.temp
  A_mul_B!(f.temp, f.Q, y)
  fy = 0.5*vecdot(y, f.temp) + vecdot(y, f.q)
  return fy
end

function factor_step!(f::QuadraticDirect{R, M, V, F}, gamma::R) where {R, M <: DenseMatrix{R}, V, F}
  f.gamma = gamma;
  f.fact = cholfact(f.Q + I/gamma, :L);
end

function factor_step!(f::QuadraticDirect{R, M, V, F}, gamma::R) where {R, I, M <: SparseMatrixCSC{R, I}, V, F}
  f.gamma = gamma;
  f.fact = ldltfact(f.Q; shift = 1/gamma);
end

function gradient!(y::AbstractArray{R}, f::QuadraticDirect{R, M, V, F}, x::AbstractArray{R}) where {R, M, V, F}
  A_mul_B!(y, f.Q, x)
  y .+= f.q
  return 0.5*(vecdot(x, y) + vecdot(x, f.q))
end

function prox_naive(f::QuadraticDirect, x, gamma=1.0)
  y = (gamma*f.Q + I)\(x - gamma*f.q)
  fy = 0.5*vecdot(y, f.Q*y) + vecdot(y, f.q)
  return y, fy
end
