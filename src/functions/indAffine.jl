# indicator of an affine set

export IndAffine

"""
**Indicator of an affine subspace**

    IndAffine(A, b)

If `A` is a matrix (dense or sparse) and `b` is a vector, returns the indicator function of the set
```math
S = \\{x : Ax = b\\}.
```
If `A` is a vector and `b` is a scalar, returns the indicator function of the set
```math
S = \\{x : \\langle A, x \\rangle = b\\}.
```
By default, a direct method (matrix factorization) is used to evaluate `prox!`.
If `iterative=true`, then `prox!` is evaluated approximately using an iterative method instead.
"""

### ABSTRACT TYPE

abstract type IndAffine <: ProximableFunction end

is_affine(f::IndAffine) = true
is_generalized_quadratic(f::IndAffine) = true

fun_name(f::IndAffine) = "Indicator of an affine subspace"

### CONSTRUCTORS

function IndAffine(A::M, b::V; iterative=false) where {M, V}
  if iterative == false
    IndAffineDirect(A, b)
  else
    error("not yet implemented")
  end
end

### INCLUDE CONCRETE TYPES

include("indAffineDirect.jl")
include("indAffineIterative.jl")
