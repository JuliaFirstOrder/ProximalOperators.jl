# indicator of an affine set

using LinearAlgebra
using SparseArrays
using SuiteSparse

export IndAffine

### ABSTRACT TYPE

abstract type IndAffine end

is_affine(f::Type{<:IndAffine}) = true
is_generalized_quadratic(f::Type{<:IndAffine}) = true

fun_name(f::IndAffine) = "Indicator of an affine subspace"

### CONSTRUCTORS

"""
    IndAffine(A, b; iterative=false)

If `A` is a matrix (dense or sparse) and `b` is a vector, return the indicator function of the affine set
```math
S = \\{x : Ax = b\\}.
```
If `A` is a vector and `b` is a scalar, return the indicator function of the set
```math
S = \\{x : \\langle A, x \\rangle = b\\}.
```
By default, a direct method (QR factorization of matrix `A'`) is used to evaluate `prox!`.
If `iterative=true`, then `prox!` is evaluated approximately using an iterative method instead.
"""
function IndAffine(A::M, b::V; iterative=false) where {M, V}
    if iterative == false
        IndAffineDirect(A, b)
    else
        IndAffineIterative(A, b)
    end
end

### INCLUDE CONCRETE TYPES

using LinearAlgebra
using SparseArrays
using SuiteSparse

include("indAffineDirect.jl")
include("indAffineIterative.jl")

function prox_naive(f::IndAffine, x, gamma)
    y = x + f.A'*((f.A*f.A')\(f.b - f.A*x))
    return y, real(eltype(x))(0)
end
