# quadratic function

export Quadratic

### ABSTRACT TYPE

abstract type Quadratic <: ProximableFunction end

is_convex(f::Quadratic) = true
is_smooth(f::Quadratic) = true
is_quadratic(f::Quadratic) = true

fun_name(f::Quadratic) = "Quadratic function"

### CONSTRUCTORS

"""
**Quadratic function**

    Quadratic(Q, q; iterative=false)

For a matrix `Q` (dense or sparse, symmetric and positive semidefinite) and a vector `q`, returns the function
```math
f(x) = \\tfrac{1}{2}\\langle Qx, x\\rangle + \\langle q, x \\rangle.
```
By default, a direct method (based on Cholesky factorization) is used to evaluate `prox!`.
If `iterative=true`, then `prox!` is evaluated approximately using an iterative method instead.
"""
function Quadratic(Q::M, q::V; iterative=false) where {M, V}
    if iterative == false
        QuadraticDirect(Q, q)
    else
        QuadraticIterative(Q, q)
    end
end

### INCLUDE CONCRETE TYPES

include("quadraticDirect.jl")
include("quadraticIterative.jl")
