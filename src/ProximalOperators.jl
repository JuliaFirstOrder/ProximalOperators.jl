# ProximalOperators.jl - library of commonly used functions in optimization, and associated proximal mappings and gradients

__precompile__()

module ProximalOperators

const RealOrComplex{T<:Real} = Union{T, Complex{T}}
const HermOrSym{T, S} = Union{Hermitian{T, S}, Symmetric{T, S}}

export ProximableFunction
export prox, prox!
export gradient!

import Base: gradient

abstract type ProximableFunction end

# Utilities

include("utilities/cg.jl")
include("utilities/deep.jl")
include("utilities/symmetricpacked.jl")

# Basic functions

include("functions/elasticNet.jl")
include("functions/hingeLoss.jl")
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
include("functions/indPSD.jl")
include("functions/indSimplex.jl")
include("functions/indSOC.jl")
include("functions/indSphereL2.jl")
include("functions/indZero.jl")
include("functions/leastSquares.jl")
include("functions/linear.jl")
include("functions/logBarrier.jl")
include("functions/maximum.jl")
include("functions/normL2.jl")
include("functions/normL1.jl")
include("functions/normL21.jl")
include("functions/normL0.jl")
include("functions/nuclearNorm.jl")
include("functions/quadratic.jl")
include("functions/sqrNormL2.jl")
include("functions/sumPositive.jl")

# Calculus rules

include("calculus/conjugate.jl")
include("calculus/epicompose.jl")
include("calculus/distL2.jl")
include("calculus/moreauEnvelope.jl")
include("calculus/postcompose.jl")
include("calculus/precomposeDiagonal.jl")
include("calculus/precomposeGramDiagonal.jl")
include("calculus/regularize.jl")
include("calculus/separableSum.jl")
include("calculus/slicedSeparableSum.jl")
include("calculus/sqrDistL2.jl")
include("calculus/tilt.jl")
include("calculus/translate.jl")

# Functions obtain from basic + calculus

include("functions/indExp.jl")
include("functions/sumLargest.jl")
include("functions/normLinf.jl")

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
  prox(f::ProximableFunction, x, γ=1.0)

Computes the proximal point of `x` with respect to function `f`
and parameter `γ > 0`, that is

  y = argmin_z { f(z) + 1/(2γ)||z-x||² }

and returns `y` and `f(y)`.
"""

function prox(f::ProximableFunction, x, gamma=1.0)
  y = deepsimilar(x)
  fy = prox!(y, f, x, gamma)
  return y, fy
end

"""
  prox!(y, f::ProximableFunction, x, γ=1.0)

Computes the proximal point of `x` with respect to function `f`
and parameter `γ > 0`, and writes the result in `y`. Returns `f(y)`.
"""

prox!

"""
  gradient(f::ProximableFunction, x, γ=1.0)

Computes the gradient of `f` at `x`: be it `g`, the function returns `g` and `f(x)`.
"""

function gradient(f::ProximableFunction, x)
	y = deepsimilar(x)
	fx = gradient!(y, f, x)
	return y, fx
end

"""
  gradient!(y, f::ProximableFunction, x)

Computes the gradient of `f` at `x` and writes it to `y`. Returns `f(x)`.
"""

gradient!

end
