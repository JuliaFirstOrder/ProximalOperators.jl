# ProximalOperators.jl - library of commonly used functions in optimization, and associated proximal mappings and gradients

module ProximalOperators

using LinearAlgebra
import ProximalCore: prox, prox!, gradient, gradient!
import ProximalCore: is_convex, is_generalized_quadratic

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
include("utilities/linops.jl")
include("utilities/symmetricpacked.jl")
include("utilities/uniformarrays.jl")
include("utilities/normdiff.jl")
include("utilities/traits.jl")

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
include("functions/totalVariation1D.jl")

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

end
