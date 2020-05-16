# least squares penalty

export LeastSquares

### ABSTRACT TYPE

abstract type LeastSquares <: ProximableFunction end

is_smooth(f::LeastSquares) = true
is_quadratic(f::LeastSquares) = true

fun_name(f::LeastSquares) = "Least squares penalty"

### CONSTRUCTORS

"""
**Least squares penalty**

    LeastSquares(A, b, λ=1.0; iterative=false)

For a matrix `A`, a vector `b` and a scalar `λ`, returns the function
```math
f(x) = \\tfrac{\\lambda}{2}\\|Ax - b\\|^2.
```
By default, a direct method (based on Cholesky factorization) is used to evaluate `prox!`.
If `iterative=true`, then `prox!` is evaluated approximately using an iterative method instead.
"""
function LeastSquares(A::M, b::V, lam::R=R(1); iterative=false) where {R <: Real, RC <: RealOrComplex{R}, V <: AbstractArray{RC}, M}
    if iterative == false
        LeastSquaresDirect(A, b, lam)
    else
        LeastSquaresIterative(A, b, lam)
    end
end

infer_shape_of_x(A::AbstractMatrix, ::AbstractVector) = (size(A, 2), )
infer_shape_of_x(A::AbstractMatrix, b::AbstractMatrix) = (size(A, 2), size(b, 2))

### INCLUDE CONCRETE TYPES

include("leastSquaresDirect.jl")
include("leastSquaresIterative.jl")
