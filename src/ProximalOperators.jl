# ProximalOperators.jl - library of commonly used functions in optimization, and associated proximal mappings and gradients

__precompile__()

module ProximalOperators

using IterativeSolvers
using OSQP

const RealOrComplex{T <: Real} = Union{T, Complex{T}}
const HermOrSym{T, S} = Union{Hermitian{T, S}, Symmetric{T, S}}

export ProximableFunction
export prox, prox!
export gradient!

import Base: gradient

abstract type ProximableFunction end

# Utilities

include("utilities/deep.jl")
include("utilities/linops.jl")
include("utilities/symmetricpacked.jl")
include("utilities/uniformarrays.jl")
include("utilities/vecnormdiff.jl")

# Basic functions

include("functions/elasticNet.jl")
include("functions/huberLoss.jl")
include("functions/indAffine.jl")
include("functions/indBallL0.jl")
include("functions/indBallL1.jl")
include("functions/indBallL2.jl")
include("functions/indBallRank.jl")
include("functions/indBinary.jl")
include("functions/indBox.jl")
include("functions/indFree.jl")
include("functions/indHalfspace.jl")
include("functions/indNonnegative.jl")
include("functions/indNonpositive.jl")
include("functions/indPoint.jl")
include("functions/indPolyhedral.jl")
include("functions/indPSD.jl")
include("functions/indSimplex.jl")
include("functions/indSOC.jl")
include("functions/indSphereL2.jl")
include("functions/indZero.jl")
include("functions/leastSquares.jl")
include("functions/linear.jl")
include("functions/logBarrier.jl")
include("functions/logisticLoss.jl")
include("functions/maximum.jl")
include("functions/normL2.jl")
include("functions/normL1.jl")
include("functions/normL21.jl")
include("functions/normL0.jl")
include("functions/nuclearNorm.jl")
include("functions/quadratic.jl")
include("functions/sqrNormL2.jl")
include("functions/sumPositive.jl")
include("functions/indGraph.jl")
include("functions/sqrHingeLoss.jl")
include("functions/crossEntropy.jl")

# Calculus rules

include("calculus/conjugate.jl")
# include("calculus/epicompose.jl")
include("calculus/distL2.jl")
include("calculus/moreauEnvelope.jl")
include("calculus/postcompose.jl")
include("calculus/precompose.jl")
include("calculus/precomposeDiagonal.jl")
include("calculus/regularize.jl")
include("calculus/separableSum.jl")
include("calculus/slicedSeparableSum.jl")
include("calculus/sqrDistL2.jl")
include("calculus/tilt.jl")
include("calculus/translate.jl")

# Functions obtained from basic + calculus

include("functions/hingeLoss.jl")
include("functions/indExp.jl")
include("functions/normLinf.jl")
include("functions/sumLargest.jl")

function Base.show(io::IO, f::ProximableFunction)
  println(io, "description : ", fun_name(f))
  println(io, "domain      : ", fun_dom(f))
  println(io, "expression  : ", fun_expr(f))
  print(  io, "parameters  : ", fun_params(f))
end

fun_name(  f) = "n/a"
fun_dom(   f) = "n/a"
fun_expr(  f) = "n/a"
fun_params(f) = "n/a"

is_prox_accurate(f::ProximableFunction) = true
is_separable(f::ProximableFunction) = false
is_convex(f::ProximableFunction) = false
is_singleton(f::ProximableFunction) = false
is_cone(f::ProximableFunction) = false
is_affine(f::ProximableFunction) = is_singleton(f)
is_set(f::ProximableFunction) = is_cone(f) || is_affine(f)
is_smooth(f::ProximableFunction) = false
is_quadratic(f::ProximableFunction) = false
is_generalized_quadratic(f::ProximableFunction) = is_quadratic(f) || is_affine(f)
is_strongly_convex(f::ProximableFunction) = false

"""
**Proximal mapping**

    y, fy = prox(f, x, γ=1.0)

Computes
```math
y = \\mathrm{prox}_{\\gamma f}(x) = \\arg\\min_z \\left\\{ f(z) + \\tfrac{1}{2\\gamma}\\|z-x\\|^2 \\right\\}.
```
Return values:
* `y`: the proximal point ``y``
* `fy`: the value ``f(y)``
"""

function prox(f::ProximableFunction, x, gamma=1.0)
  y = deepsimilar(x)
  fy = prox!(y, f, x, gamma)
  return y, fy
end

"""
**Proximal mapping (in-place)**

    fy = prox!(y, f, x, γ=1.0)

Computes
```math
y = \\mathrm{prox}_{\\gamma f}(x) = \\arg\\min_z \\left\\{ f(z) + \\tfrac{1}{2\\gamma}\\|z-x\\|^2 \\right\\}.
```
The resulting point ``y`` is written to the (pre-allocated) array `y`, which must have the same shape/size as `x`.

Return values:
* `fy`: the value ``f(y)``
"""

prox!

"""
**Gradient mapping**

    gradfx, fx = gradient(f, x)

Computes the gradient (and value) of ``f`` at ``x``. If ``f`` is only *subdifferentiable* at ``x``, then return a subgradient instead.

Return values:
* `gradfx`: the (sub)gradient of ``f`` at ``x``
* `fx`: the value ``f(x)``
"""

function gradient(f::ProximableFunction, x)
	y = deepsimilar(x)
	fx = gradient!(y, f, x)
	return y, fx
end

"""
**Gradient mapping (in-place)**

    gradient!(gradfx, f, x)

Writes ``\\nabla f(x)`` to `gradfx`, which must be pre-allocated and have the same shape/size as `x`. If ``f`` is only *subdifferentiable* at ``x``, then writes a subgradient instead.

Return values:
* `fx`: the value ``f(x)``
"""

gradient!

end
