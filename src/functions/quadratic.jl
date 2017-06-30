# quadratic function

export Quadratic

"""
**Quadratic function**

    Quadratic(Q, q)

For a matrix `Q` (dense or sparse, symmetric and positive definite) and a vector `q`, returns the function
```math
f(x) = \\tfrac{1}{2}\\langle Qx, x\\rangle + \\langle q, x \\rangle.
```
"""

type Quadratic{R <: Real, RC <: Union{R, Complex{R}}, M <: AbstractMatrix{RC}, V <: AbstractVector{RC}, F <: Factorization} <: ProximableFunction
  Q::M
  q::V
  gamma::R
  iter::Bool
  fact::F
  function Quadratic{R, RC, M, V, F}(Q::M, q::V, iter::Bool) where {R <: Real, RC <: Union{R, Complex{R}}, M <: AbstractMatrix{RC}, V <: AbstractVector{RC}, F <: Factorization}
    if size(Q, 1) != size(Q, 2) || length(q) != size(Q, 2)
      error("Q must be squared and q must be compatible with Q")
    end
    new(Q, q, -1, iter)
  end
end

is_smooth(f::Quadratic) = true
is_quadratic(f::Quadratic) = true
is_prox_accurate(f::Quadratic) = !f.iter

function Quadratic(Q::M, q::V, iter::Bool=false) where {R <: Real, RC <: Union{R, Complex{R}}, I <: Integer, M <: SparseMatrixCSC{RC, I}, V <: AbstractVector{RC}}
  Quadratic{R, RC, M, V, SparseArrays.CHOLMOD.Factor{RC}}(Q, q, iter)
end

function Quadratic(Q::M, q::V, iter::Bool=false) where {R <: Real, RC <: Union{R, Complex{R}}, M <: DenseMatrix{RC}, V <: AbstractVector{RC}}
  Quadratic{R, RC, M, V, LinAlg.Cholesky{RC}}(Q, q, iter)
end

function (f::Quadratic){R <: RealOrComplex}(x::AbstractArray{R})
  return 0.5*vecdot(x, f.Q*x) + vecdot(x, f.q)
end

function prox!{R, RC, M, V, F}(y::AbstractArray{RC}, f::Quadratic{R, RC, M, V, F}, x::AbstractArray{RC}, gamma::R=1.0)
  if f.iter == false
    if gamma != f.gamma
      factor_step!(f, gamma)
    end
    y .= f.fact\(x/gamma - f.q)
  else
    cg!(y, f.Q, 1.0/gamma, x/gamma - f.q)
  end
  fy = 0.5*vecdot(y, f.Q*y) + vecdot(y, f.q)
  return fy
end

function factor_step!{R, RC, I <: Integer, M <: SparseMatrixCSC{RC, I}, V, F}(f::Quadratic{R, RC, M, V, F}, gamma::R)
  f.gamma = gamma;
  f.fact = ldltfact(f.Q; shift = 1/gamma);
end

function factor_step!{R, RC, M <: DenseMatrix{RC}, V, F}(f::Quadratic{R, RC, M, V, F}, gamma::R)
  f.gamma = gamma;
  f.fact = cholfact(f.Q + I/gamma);
end

function gradient!{R, RC, M, V, F}(y::AbstractArray{RC}, f::Quadratic{R, RC, M, V, F}, x::AbstractArray{RC})
  A_mul_B!(y, f.Q, x)
  y .+= f.q
  return 0.5*(vecdot(x, y) + vecdot(x, f.q))
end

fun_name(f::Quadratic) = "Quadratic function"
fun_expr(f::Quadratic) = "x ↦ (1/2)*(x'Qx) + q'x"

function prox_naive(f::Quadratic, x, gamma=1.0)
  y = (gamma*f.Q + I)\(x - gamma*f.q)
  fy = 0.5*vecdot(y, f.Q*y) + vecdot(y, f.q)
  return y, fy
end
