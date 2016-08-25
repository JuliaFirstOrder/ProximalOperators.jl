# indicator of a halfspace

"""
  IndHalfspace(a::Array{Float64}, b::Float64)

Returns the function `g = ind{x : <a,x> ⩽ b}`.
"""

immutable IndHalfspace <: IndicatorConvex
  a::Array{Float64}
  b::Float64
  function IndHalfspace(a::Array{Float64}, b::Float64)
    norma = vecnorm(a)
    new(a/norma, b/norma)
  end
end

function call(f::IndHalfspace, x::Array{Float64})
  s = vecdot(f.a,x)-f.b
  if s <= 1e-14 return 0.0 end
  return +Inf
end

function prox(f::IndHalfspace, x::Array{Float64}, gamma::Float64=1.0)
  s = vecdot(f.a,x)-f.b
  if s <= 0 return (x, 0.0) end
  return (x - s*f.a, 0.0)
end

fun_name(f::IndHalfspace) = "indicator of a halfspace"
fun_type(f::IndHalfspace) = "R^n → R ∪ {+∞}"
fun_expr(f::IndHalfspace) = "x ↦ 0 if <a,x> ⩽ b, +∞ otherwise"
fun_params(f::IndHalfspace) =
  string( "a = ", typeof(f.a), " of size ", size(f.a), ", ",
          "b = $(f.b)")
