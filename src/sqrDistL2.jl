# squared Euclidean distance from a set

immutable SqrDistL2{T <: Union{Float64,Array{Float64}}} <: ProximableFunction
  ind::IndicatorConvex
  lambda::T
  SqrDistL2(ind::IndicatorConvex, lambda) =
    any(lambda .< 0) ? error("coefficients in λ must be nonnegative") : new(ind, lambda)
end

SqrDistL2(ind::IndicatorConvex, lambda::Float64=1.0) = SqrDistL2{Float64}(ind, lambda)

SqrDistL2(ind::IndicatorConvex, lambda::Array{Float64}) = SqrDistL2{Array{Float64}}(ind, lambda)

function call(f::SqrDistL2{Float64}, x::Array)
  p, = prox(f.ind, 1.0, x)
  return (f.lambda/2)*vecnorm(x-p)^2
end

function prox(f::SqrDistL2{Float64}, gamma::Float64, x::Array)
  p, = prox(f.ind, 1.0, x)
  sqrd = (f.lambda/2)*vecnorm(x-p)^2
  gamlam = f.lambda*gamma
  return 1/(1+gamlam)*x + gamlam/(1+gamlam)*p, sqrd/(1+gamlam)^2
end
