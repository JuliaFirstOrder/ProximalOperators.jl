export IndPolyhedral

abstract type IndPolyhedral <: ProximableFunction end

is_convex(::IndPolyhedral) = true
is_set(::IndPolyhedral) = true

"""
**Indicator of a polyhedral set**

    IndPolyhedral([l,] A, [u, xmin, xmax])

Returns the indicator function of the polyhedral set:
```math
S = \\{ x : x_\\min \\leq x \\leq x_\\max, l \\leq Ax \\leq u \\}.
```
Matrix `A` is a mandatory argument; when any of the bounds is not provided,
it is assumed to be (plus or minus) infinity.
"""
function IndPolyhedral(args...; solver=:osqp)
    if solver == :osqp
        IndPolyhedralOSQP(args...)
    elseif solver == :qpdas
        IndPolyhedralQPDAS(args...)
    else
        error("unknown solver")
    end
end

# including concrete types

include("indPolyhedralOSQP.jl")
include("indPolyhedralQPDAS.jl")
