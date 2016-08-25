immutable ScaleTranslate{T <: ProximableFunction} <: ProximableFunction
  f::T
  a::Float64
  b::Array
end

call(g::ScaleTranslate, x) = g.f(g.a*x + g.b)

function prox(g::ScaleTranslate, x, gamma::Float64=1.0)
  p, v = prox(g.f, g.a*x + g.b, (g.a^2)*gamma)
  return (p-g.b)/g.a, v
end
