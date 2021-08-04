# ProximalOperators.jl - library of commonly used functions in optimization, and associated proximal mappings and gradients

module ProximalOperators

using LinearAlgebra

const RealOrComplex{R <: Real} = Union{R, Complex{R}}
const HermOrSym{T, S} = Union{Hermitian{T, S}, Symmetric{T, S}}
const RealBasedArray{R} = AbstractArray{C, N} where {C <: RealOrComplex{R}, N}
const TupleOfArrays{R} = Tuple{RealBasedArray{R}, Vararg{RealBasedArray{R}}}
const ArrayOrTuple{R} = Union{RealBasedArray{R}, TupleOfArrays{R}}
const TransposeOrAdjoint{M} = Union{Transpose{C,M} where C, Adjoint{C,M} where C}
const Maybe{T} = Union{T, Nothing}

export prox, prox!, gradient, gradient!

# Utilities

include("utilities/approx_inequality.jl")
include("utilities/tuples.jl")
include("utilities/linops.jl")
include("utilities/symmetricpacked.jl")
include("utilities/uniformarrays.jl")
include("utilities/normdiff.jl")

# Basic functions

include("functions/cubeNormL2.jl")
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
include("functions/indGraph.jl")
include("functions/indHalfspace.jl")
include("functions/indHyperslab.jl")
include("functions/indNonnegative.jl")
include("functions/indNonpositive.jl")
include("functions/indPoint.jl")
include("functions/indPolyhedral.jl")
include("functions/indPSD.jl")
include("functions/indSimplex.jl")
include("functions/indSOC.jl")
include("functions/indSphereL2.jl")
include("functions/indStiefel.jl")
include("functions/indZero.jl")
include("functions/leastSquares.jl")
include("functions/linear.jl")
include("functions/logBarrier.jl")
include("functions/logisticLoss.jl")
include("functions/normL0.jl")
include("functions/normL1.jl")
include("functions/normL2.jl")
include("functions/normL21.jl")
include("functions/normL1plusL2.jl")
include("functions/nuclearNorm.jl")
include("functions/quadratic.jl")
include("functions/sqrNormL2.jl")
include("functions/sumPositive.jl")
include("functions/sqrHingeLoss.jl")
include("functions/crossEntropy.jl")
include("functions/TotalVariation1D.jl")
# Calculus rules

include("calculus/conjugate.jl")
include("calculus/epicompose.jl")
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
include("calculus/sum.jl")
include("calculus/pointwiseMinimum.jl")

# Functions obtained from basic (as special cases or using calculus rules)

include("functions/hingeLoss.jl")
include("functions/indExp.jl")
include("functions/maximum.jl")
include("functions/normLinf.jl")
include("functions/sumLargest.jl")

is_prox_accurate(_) = true
is_separable(_) = false
is_convex(_) = false
is_concave(_) = false
is_singleton(_) = false
is_cone(_) = false
is_affine(f) = is_singleton(f)
is_set(f) = is_cone(f) || is_affine(f)
is_positively_homogeneous(f) = is_cone(f)
is_support(f) = is_convex(f) && is_positively_homogeneous(f)
is_smooth(_) = false
is_quadratic(_) = false
is_generalized_quadratic(f) = is_quadratic(f) || is_affine(f)
is_strongly_convex(_) = false

"""
**Proximal mapping**

    y, fy = prox(f, x, gamma=1)

Computes
```math
y = \\mathrm{prox}_{\\gamma f}(x) = \\arg\\min_z \\left\\{ f(z) + \\tfrac{1}{2\\gamma}\\|z-x\\|^2 \\right\\}.
```
Return values:
* `y`: the proximal point ``y``
* `fy`: the value ``f(y)``
"""
function prox(f, x, gamma=1)
    y = similar(x)
    fy = prox!(y, f, x, gamma)
    return y, fy
end

"""
**Proximal mapping (in-place)**

    fy = prox!(y, f, x, gamma=1)

Computes
```math
y = \\mathrm{prox}_{\\gamma f}(x) = \\arg\\min_z \\left\\{ f(z) + \\tfrac{1}{2\\gamma}\\|z-x\\|^2 \\right\\}.
```
The resulting point ``y`` is written to the (pre-allocated) array `y`, which must have the same shape/size as `x`.

Return values:
* `fy`: the value ``f(y)``
"""
prox!(y, f, x) = prox!(y, f, x, 1)

"""
**Gradient mapping**

    gradfx, fx = gradient(f, x)

Computes the gradient (and value) of ``f`` at ``x``. If ``f`` is only *subdifferentiable* at ``x``, then return a subgradient instead.

Return values:
* `gradfx`: the (sub)gradient of ``f`` at ``x``
* `fx`: the value ``f(x)``
"""
function gradient(f, x)
    y = similar(x)
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
