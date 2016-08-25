immutable ScaleTranslate{T <: ProximableFunction} <: ProximableFunction
  f::T
  a::Float64
  b::Array
end

call(g::ScaleTranslate, x) = g.f(g.a*x + g.b)

function prox(g::ScaleTranslate, gamma::Float64, x)
  p, v = prox(g.f, (g.a^2)*gamma, g.a*x + g.b)
  return (p-g.b)/g.a, v
end
