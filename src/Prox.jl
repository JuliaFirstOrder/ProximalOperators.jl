# Prox.jl - library of nonsmooth functions and associated proximal mappings

__precompile__()

module Prox

typealias RealOrComplexArray Union{Array{Float64}, Array{Complex{Float64}}}
typealias RealOrComplexVector Union{Array{Float64,1}, Array{Complex{Float64},1}}
typealias RealOrComplexMatrix Union{Array{Float64,2}, Array{Complex{Float64},2}}

export prox

export ProximableFunction,
       IndAffine,
       IndBallInf,
       IndBallL0,
       IndBallL2,
       IndBallL20,
       IndBallRank,
       IndBox,
       IndHalfspace,
       IndNonnegative,
       IndSimplex,
       IndSOC,
       ElasticNet,
       NormL0,
       NormL1,
       NormL2,
       NormL21,
       SqrNormL2,
       DistL2,
       SqrDistL2

# This hierarchy of abstract types (or a similar one) may be useful.
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
include("functions/normL2.jl")
include("functions/normL1.jl")
include("functions/normL21.jl")
include("functions/normL0.jl")
include("functions/indAffine.jl")
include("functions/indBallL0.jl")
include("functions/indBallL2.jl")
include("functions/indBallRank.jl")
include("functions/indBox.jl")
include("functions/indSOC.jl")
include("functions/indHalfspace.jl")
include("functions/indSimplex.jl")
include("functions/sqrDistL2.jl")
include("functions/sqrNormL2.jl")

function call(f::ProximableFunction, x)
  error("call is not implemented for type ", typeof(f))
end

"""
  prox(f::ProximableFunction, γ::Float64, x::Array)

Computes the proximal point of `x` with respect to function `f`
and parameter `γ > 0`, that is

  y = argmin_z { f(z) + 1/(2γ)||z-x||^2 }

and returns `y` and `f(y)`.
"""

function prox(f::ProximableFunction, x, gamma::Float64=1.0)
  error("prox is not implemented for type ", typeof(f))
end

function Base.show(io::IO, f::ProximableFunction)
  println(io, "description : ", fun_name(f))
  println(io, "type        : ", fun_type(f))
  println(io, "expression  : ", fun_expr(f))
  print(  io, "parameters  : ", fun_params(f))
end

fun_name(  f::ProximableFunction) = "n/a"
fun_type(  f::ProximableFunction) = "n/a"
fun_expr(  f::ProximableFunction) = "n/a"
fun_params(f::ProximableFunction) = "n/a"

end
