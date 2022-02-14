# least squares penalty

export LeastSquares

### ABSTRACT TYPE

abstract type LeastSquares end

is_smooth(::Type{<:LeastSquares}) = true
is_generalized_quadratic(::Type{<:LeastSquares}) = true

### CONSTRUCTORS

"""
    LeastSquares(A, b, λ=1.0; iterative=false)

For a matrix `A`, a vector `b` and a scalar `λ`, return the function
```math
f(x) = \\tfrac{\\lambda}{2}\\|Ax - b\\|^2.
```
By default, a direct method (based on Cholesky factorization) is used to evaluate `prox!`.
If `iterative=true`, then `prox!` is evaluated approximately using an iterative method instead.
"""
function LeastSquares(A, b, lam=1; iterative=false)
    if iterative == false
        LeastSquaresDirect(A, b, lam)
    else
        LeastSquaresIterative(A, b, lam)
    end
end

infer_shape_of_x(A, ::AbstractVector) = (size(A, 2), )
infer_shape_of_x(A, b::AbstractMatrix) = (size(A, 2), size(b, 2))

### INCLUDE CONCRETE TYPES

include("leastSquaresDirect.jl")
include("leastSquaresIterative.jl")
