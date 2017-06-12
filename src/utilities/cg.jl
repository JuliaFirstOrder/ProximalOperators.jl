# conjugate-gradient method for solving Ax = b, with A spd

"""
  cg!(x, A, b, [tol=1e-6, M=1.0])

Computes an approximate solution to the system `Ax = b` and writes it in `x`.
Optional parameter `tol` is the desired precision, and `M` is a preconditioner: with `M = A` the algorithms should terminate in 1 iteration.
"""
function cg!(x::Vx, A::MA, b::Vb, tol=1e-6, M=1.0) where {R <: Real, C <: Union{R, Complex{R}}, Vx <: AbstractVector{C}, MA <: AbstractMatrix{C}, Vb <: AbstractVector{C}}
  x .= 0
  Ap = similar(b)
  r = b - A*x
  z = solve(M, r)
  p = copy(z)
  rzold = dot(r, z)
  k = 0
  for k = 1:length(b)
    A_mul_B!(Ap, A, p)
    alpha = rzold/dot(p, Ap)
    x .+= alpha*p
    r .-= alpha*Ap
    if norm(r) < tol
      break
    end
    solve!(z, M, r)
    rznew = dot(r, z)
    p .*= rznew/rzold
    p .+= z
    rzold = rznew
  end
  return k
end

########################################################################
# Utility functions to solve simple, spd linear systems:
# - These are used for preconditioning purposes
########################################################################

function solve(A, b)
  x = similar(b)
  solve!(x, A, b)
  return x
end

function solve!(x::Vx, A::C, b::Vb) where {R <: Real, C <: Union{R, Complex{R}}, Vx <: AbstractVector{C}, Vb <: AbstractVector{C}}
  x .= A .* b
end

function solve!(x::Vx, A::VA, b::Vb) where {R <: Real, C <: Union{R, Complex{R}}, Vx <: AbstractVector{C}, VA <: AbstractVector{C}, Vb <: AbstractVector{C}}
  x .= b ./ A
end
