# quadratic function (iterative prox)

export QuadraticIterative

"""
**Quadratic function (iterative evaluation of prox)**

    QuadraticIterative(Q, q)

For a matrix `Q` (dense or sparse, symmetric and positive definite) and a vector `q`, returns the function
```math
f(x) = \\tfrac{1}{2}\\langle Qx, x\\rangle + \\langle q, x \\rangle.
```
Differently from `Quadratic`, in this case the `prox` operation is evaluated (inexactly) using an iterative method.
"""

immutable QuadraticIterative{R <: Real, RC <: Union{R, Complex{R}}, M <: AbstractMatrix{RC}, V <: AbstractVector{RC}} <: ProximableFunction
  Q::M
  q::V
  function QuadraticIterative{R, RC, M, V}(Q::M, q::V) where {R <: Real, RC <: Union{R, Complex{R}}, M <: AbstractMatrix{RC}, V <: AbstractVector{RC}}
    if size(Q, 1) != size(Q, 2) || length(q) != size(Q, 2)
      error("Q must be squared and q must be compatible with Q")
    end
    new(Q, q)
  end
end

is_smooth(f::QuadraticIterative) = true
is_quadratic(f::QuadraticIterative) = true
is_prox_accurate(f::QuadraticIterative) = false

function QuadraticIterative(Q::M, q::V) where {R <: Real, RC <: Union{R, Complex{R}}, M <: AbstractMatrix{RC}, V <: AbstractVector{RC}}
  QuadraticIterative{R, RC, M, V}(Q, q)
end

function (f::QuadraticIterative){R <: RealOrComplex}(x::AbstractArray{R})
  return 0.5*vecdot(x, f.Q*x) + vecdot(x, f.q)
end

function prox!{R, RC, M, V}(y::AbstractArray{RC}, f::QuadraticIterative{R, RC, M, V}, x::AbstractArray{RC}, gamma::R=1.0)
  cg!(y, f.Q, 1.0/gamma, x/gamma - f.q)
  fy = 0.5*vecdot(y, f.Q*y) + vecdot(y, f.q)
  return fy
end

function gradient!{R, RC, M, V}(y::AbstractArray{RC}, f::QuadraticIterative{R, RC, M, V}, x::AbstractArray{RC})
  A_mul_B!(y, f.Q, x)
  y .+= f.q
  return 0.5*(vecdot(x, y) + vecdot(x, f.q))
end

fun_name(f::QuadraticIterative) = "Quadratic function (iterative prox)"
fun_expr(f::QuadraticIterative) = "x ↦ (1/2)*(x'Qx) + q'x"

function prox_naive(f::QuadraticIterative, x, gamma=1.0)
  y = (gamma*f.Q + I)\(x - gamma*f.q)
  fy = 0.5*vecdot(y, f.Q*y) + vecdot(y, f.q)
  return y, fy
end
