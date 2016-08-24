# Conjugate function: can we implement call somehow?
# I guess we would need to invert the subgradient of f, i.e., to have the
# subgradient of f_star... Maybe one could use AD?
# The prox is ok anyway, and in the end that's what is needed in algorithms.

"""
  Conjugate(f::ProximableFunction)

Returns the Fenchel conjugate of a given convex function `f`.
"""

immutable Conjugate <: ProximableFunction
  f::ProximableFunction
end

function prox(g::Conjugate, gamma::Float64, x::Array)
  z, v = prox(g.f, 1/gamma, x/gamma)
  p = x - gamma*z
  return p, vecdot(p,z) - v
end

fun_name(g::Conjugate) = string("conjugate of ", fun_name(g.f))
fun_type(g::Conjugate) = fun_type(g.f)
fun_expr(g::Conjugate) = string("conjugate of ", fun_expr(g.f))
fun_params(g::Conjugate) = fun_params(g.f)
