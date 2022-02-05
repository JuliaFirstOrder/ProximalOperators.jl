export IndPolyhedral

abstract type IndPolyhedral end

is_convex(::Type{<:IndPolyhedral}) = true
is_set(::Type{<:IndPolyhedral}) = true

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
    else
        error("unknown solver")
    end
end

# including concrete types

include("indPolyhedralOSQP.jl")
