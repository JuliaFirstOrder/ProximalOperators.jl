# Prox.jl - library of nonsmooth functions and associated proximal mappings

__precompile__()

module Prox

using Compat
import Compat.String

typealias RealOrComplex Union{Real, Complex}
typealias HermOrSym{T,S} Union{Hermitian{T,S}, Symmetric{T,S}}

export prox, prox!

export ProximableFunction

export IndAffine,
       IndBallInf,
       IndBallL0,
       IndBallL1,
       IndBallL2,
       IndBallL20,
       IndBallRank,
       IndBox,
       IndHalfspace,
       IndNonnegative,
       IndNonpositive,
       IndPSD,
       IndSimplex,
       IndSOC,
       IndSphereL2,
       HingeLoss,
       ElasticNet,
       NormL0,
       NormL1,
       NormL2,
       NormL21,
       NuclearNorm,
       LogBarrier,
       SqrNormL2,
       DistL2,
       SqrDistL2

export Precomposition,
       Postcomposition,
       Conjugate

# A hierarchy of abstract types like this may be useful.
# Unfortunately Julia does not allow for multiple inheritance.
#
# ProximableFunction --+-- NormFunction
#                      |
#                      +-- IndicatorFunction -- IndicatorConvex
#

abstract ProximableFunction
abstract NormFunction <: ProximableFunction
abstract IndicatorFunction <: ProximableFunction
abstract IndicatorConvex <: IndicatorFunction

include("functions/distL2.jl")
include("functions/elasticNet.jl")
include("functions/logBarrier.jl")
include("functions/normL2.jl")
include("functions/normL1.jl")
include("functions/normL21.jl")
include("functions/normL0.jl")
include("functions/nuclearNorm.jl")
include("functions/hingeLoss.jl")
include("functions/indAffine.jl")
include("functions/indBallL0.jl")
include("functions/indBallL1.jl")
include("functions/indBallL2.jl")
include("functions/indBallRank.jl")
include("functions/indBox.jl")
include("functions/indPSD.jl")
include("functions/indSimplex.jl")
include("functions/indSOC.jl")
include("functions/indSphereL2.jl")
include("functions/indHalfspace.jl")
include("functions/sqrDistL2.jl")
include("functions/sqrNormL2.jl")

include("calculus/precomposition.jl")
include("calculus/postcomposition.jl")
include("calculus/conjugate.jl")

function Base.show(io::IO, f::ProximableFunction)
  println(io, "description : ", fun_name(f))
  println(io, "type        : ", fun_type(f))
  println(io, "expression  : ", fun_expr(f))
  print(  io, "parameters  : ", fun_params(f))
end

fun_name(  f) = "n/a"
fun_type(  f) = "n/a"
fun_expr(  f) = "n/a"
fun_params(f) = "n/a"

"""
  prox(f::ProximableFunction, x::AbstractArray, γ::Real=1.0)

Computes the proximal point of `x` with respect to function `f`
and parameter `γ > 0`, that is

  y = argmin_z { f(z) + 1/(2γ)||z-x||^2 }

and returns `y` and `f(y)`.
"""

function prox(f::ProximableFunction, x::AbstractArray, gamma::Real=1.0)
  y = similar(x)
  fy = prox!(f, x, y, gamma)
  return y, fy
end

"""
  prox!(f::ProximableFunction, x::AbstractArray, y::AbstractArray, γ::Real=1.0)

Computes the proximal point of `x` with respect to function `f`
and parameter `γ > 0`, and writes the result in `y`. Returns f(y)`.
"""

function prox!(f::ProximableFunction, x::AbstractArray, y::AbstractArray, gamma::Real=1.0)
  error(
    "prox! is not implemented for f::", typeof(f),
    ", x::", typeof(x), ", y::", typeof(y), ", gamma::", typeof(gamma)
  )
end

"""
  prox!(f::ProximableFunction, x::AbstractArray, γ::Real=1.0)

Computes the proximal point of `x` with respect to function `f`
and parameter `γ > 0` *in place*, that is

  x ← argmin_z { f(z) + 1/(2γ)||z-x||^2 }

and returns f(x)`.
"""

prox!(f::ProximableFunction, x::AbstractArray, gamma::Real=1.0) = prox!(f, x, x, gamma)

end
