# Epi-composition, also known as infimal postcomposition or image function.

# This is the dual operation to precomposition, see Rockafellar and Wets,
# "Variational Analysis", Theorem 11.23.
#
# Given a function f and a linear operator L, their epi-composition is:
#
#   g(y) = (Lf)(y) = inf_x { f(x) : Lx = y }.
#
# Plugging g directly in the definition of prox, one has:
#
#   prox_{\gamma g}(z) = argmin_y { (Lf)(y) + 1/(2\gamma)||y - z||^2 }
#     = argmin_y { inf_x { f(x) : Lx = y } + 1/(2\gamma)||y - z||^2 }
#     = L * argmin_x { f(x) + 1/(2\gamma)||Lx - z||^2 }.
#
# When L is such that L'*L = mu*Id, then this just requires prox_{\gamma f}.
#
# In some other cases the prox can be "easily" computed, such as when f is
# quadratic or extended-quadratic.
#

export Epicompose

"""
    Epicompose(L, f, [mu])

Return the epi-composition of ``f`` with ``L``, also known as infimal
postcomposition or image function. Given a function f and a linear operator L,
their epi-composition is:
```math
g(y) = (Lf)(y) = inf_x { f(x) : Lx = y }
```
This is the dual operation to precomposition, see Rockafellar and Wets,
"Variational Analysis", Theorem 11.23.

If ``mu > 0`` is specified, then ``L`` is assumed to be such that ``L'*L == mu*I``,
and the proximal operator is computable for any convex ``f``. If ``mu`` is
not specified, then ``f`` must be of ``Quadratic`` type.
"""
abstract type Epicompose end

Epicompose(L, f, mu) = EpicomposeGramDiagonal(L, f, mu)
Epicompose(L, f::T) where {T <: Quadratic} = EpicomposeQuadratic(L, f)

# TODO add properties

### INCLUDE CONCRETE TYPES

include("epicomposeGramDiagonal.jl")
include("epicomposeQuadratic.jl")
