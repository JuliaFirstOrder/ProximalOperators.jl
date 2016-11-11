# least squares penalty

"""
  LeastSquares(A, b, λ=1.0)

Returns the function `f(x) = (λ/2)⋅||Ax-b||^2`.
"""

type LeastSquares{R <: Real} <: ProximableFunction
  A::AbstractArray{R,2}
  b::AbstractArray{R,1}
  Atb::AbstractArray{R,1}
  lambda::R
  gamma::R
  S::AbstractArray{R,2}
  U::Union{LinAlg.Cholesky, SparseArrays.CHOLMOD.Factor}
  function LeastSquares(A::AbstractArray{R,2}, b::AbstractArray{R,1}, lambda::R)
    if size(A, 1) != length(b)
      error("A and b have incompatible dimensions")
    end
    if lambda <= 0
      error("lambda must be positive")
    end
    if size(A,1) >= size(A,2)
      new(A, b, A'*b, lambda, -1, A'*A)
    else
      new(A, b, A'*b, lambda, -1, A*A')
    end
  end
end

LeastSquares{R <: Real}(A::AbstractArray{R,2}, b::AbstractArray{R,1}, lambda::R=one(R)) =
  LeastSquares{R}(A, b, lambda)

@compat function (f::LeastSquares{R}){R <: Real}(x::AbstractArray{R,1})
  return (f.lambda/2)*vecnorm(f.A*x - f.b, 2)^2
end

function factor_step!{R <: Real}(f::LeastSquares{R}, gamma::R)
  # factor step, two cases: (1) tall A, (2) fat A
  lamgam = f.lambda*gamma
  if issparse(f.A)
    f.U = cholfact(f.S; shift=1.0/lamgam)
  else
    f.U = cholfact(f.S + I/lamgam)
  end
  f.gamma = gamma
end

function prox!{R <: Real}(f::LeastSquares{R}, x::AbstractArray{R,1}, y::AbstractArray{R,1}, gamma::R=1.0)
  # if gamma different from f.gamma then call factor_step!
  if gamma != f.gamma
    factor_step!(f, gamma)
  end
  lamgam = f.lambda*gamma
  # solve step, two cases: (1) tall A, (2) fat A
  q = f.Atb + x/lamgam
  if size(f.A,1) >= size(f.A,2)
    y[:] = f.U\q
  else
    y[:] = lamgam*(q - (f.A'*(f.U\(f.A*q))))
  end
  return (f.lambda/2)*norm(f.A*y-f.b, 2)^2
end

fun_name(f::LeastSquares) = "least-squares penalty"
fun_dom(f::LeastSquares) = "AbstractArray{Real,1}"
fun_expr(f::LeastSquares) = "x ↦ (λ/2)||A*x - b||^2"
fun_params(f::LeastSquares) = string("λ = $(f.lambda), A = ", typeof(f.A), " of size ", size(f.A), ", b = ", typeof(f.b), " of size ", size(f.b))

function prox_naive{R <: Real}(f::LeastSquares, x::AbstractArray{R,1}, gamma::Real=1.0)
  lamgam = f.lambda*gamma
  y = (f.A'*f.A + I/lamgam)\(f.Atb + x/lamgam)
  fy = (f.lambda/2)*norm(f.A*y-f.b)^2
  return y, fy
end
