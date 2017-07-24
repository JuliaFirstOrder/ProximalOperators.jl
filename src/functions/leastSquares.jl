# least squares penalty

export LeastSquares

"""
**Least squares penalty**

    LeastSquares(A, b, λ=1.0)

For a matrix `A`, a vector `b` and a scalar `λ`, returns the function
```math
f(x) = \\tfrac{\\lambda}{2}\\|Ax - b\\|^2.
```
By default, a direct method (matrix factorization) is used to evaluate `prox!`.
If `iterative=true`, then `prox!` is evaluated approximately using an iterative method instead.
"""

### ABSTRACT TYPE

abstract type LeastSquares <: ProximableFunction end

is_convex(f::LeastSquares) = true
is_smooth(f::LeastSquares) = true
is_quadratic(f::LeastSquares) = true

fun_name(f::LeastSquares) = "Least squares penalty"

### CONSTRUCTORS

function LeastSquares(A::M, b::V, lam::R=one(R); iterative=false) where {R <: Real, RC <: RealOrComplex{R}, V <: AbstractArray{RC}, M}
  if iterative == false
    LeastSquaresDirect(A, b, lam)
  else
    error("not yet implemented")
  end
end

### INCLUDE CONCRETE TYPES

include("leastSquaresDirect.jl")
include("leastSquaresIterative.jl")
