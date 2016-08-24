# Euclidean distance from a set

immutable DistL2 <: ProximableFunction
  ind::IndicatorConvex
  lambda::Float64
  DistL2(ind::IndicatorConvex, lambda::Float64=1.0) =
    lambda < 0 ? error("parameter λ must be nonnegative") : new(ind, lambda)
end

function call(f::DistL2, x::Array)
  p, = prox(f.ind, 1.0, x)
  return f.lambda*vecnorm(x-p)
end

function prox(f::DistL2, gamma::Float64, x::Array)
  p, = prox(f.ind, 1.0, x)
  d = vecnorm(x-p)
  gamlam = gamma*f.lambda
  if d > gamlam return (x + gamlam/d*(p-x), f.lambda*(d-gamlam)) end
  return p, 0.0
end
