export IndPolyhedral

abstract type IndPolyhedral end

is_convex(::Type{<:IndPolyhedral}) = true
is_set_indicator(::Type{<:IndPolyhedral}) = true

"""
    IndPolyhedral([l,] A, [u, xmin, xmax])

Return the indicator function of the polyhedral set:
```math
S = \\{ x : x_\\min \\leq x \\leq x_\\max, l \\leq Ax \\leq u \\}.
```
Matrix `A` is a mandatory argument; when any of the bounds is not provided,
it is assumed to be (plus or minus) infinity.

The default `solver=:osqp` backend is provided by a package extension: load it
with `using OSQP` before constructing the object, otherwise an error is raised.
"""
function IndPolyhedral(args...; solver=:osqp)
    if solver == :osqp
        IndPolyhedralOSQP(args...)
    else
        error("unknown solver")
    end
end

# IndPolyhedral: OSQP implementation
#
# The struct below is defined here so that the `solver=:osqp` dispatch above can
# refer to it, but everything that actually needs OSQP -- the constructors,
# `prox!`, the function evaluation and `prox_naive` -- lives in the package
# extension `ext/ProximalOperatorsOSQPExt.jl`, loaded once OSQP is available
# (`using OSQP`). Without OSQP loaded, constructing an `IndPolyhedralOSQP`
# (directly or via `IndPolyhedral(...; solver=:osqp)`) raises an informative
# error.

struct IndPolyhedralOSQP{R, M} <: IndPolyhedral
    l::AbstractVector{R}
    A::AbstractMatrix{R}
    u::AbstractVector{R}
    mod::M
    # Explicit inner constructor: suppresses the auto-generated 4-positional-arg
    # outer constructor, which would otherwise shadow the `(l, A, xmin, xmax)`
    # constructor added by ProximalOperatorsOSQPExt.
    IndPolyhedralOSQP{R, M}(l, A, u, mod) where {R, M} = new{R, M}(l, A, u, mod)
end

is_proximable(::Type{<:IndPolyhedralOSQP}) = false

# The real constructors are added to this function by ProximalOperatorsOSQPExt;
# this fallback only fires when OSQP is not loaded.
function IndPolyhedralOSQP(args...; kwargs...)
    error(
        "IndPolyhedralOSQP requires the OSQP package: run `using OSQP` before " *
        "constructing IndPolyhedralOSQP(...) or IndPolyhedral(...; solver=:osqp)."
    )
end
