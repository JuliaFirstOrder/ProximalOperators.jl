# squared Euclidean distance from a set

immutable SqrDistL2 <: ProximableFunction
  ind::IndicatorConvex
  lambda::Float64
  SqrDistL2(ind::IndicatorConvex, lambda::Float64=1.0) =
    lambda < 0 ? error("parameter λ must be nonnegative") : new(ind, lambda)
end

@compat function (f::SqrDistL2)(x::RealOrComplexArray)
  p, = prox(f.ind, x)
  return (f.lambda/2)*vecnorm(x-p)^2
end

function prox(f::SqrDistL2, x::RealOrComplexArray, gamma::Float64=1.0)
  p, = prox(f.ind, x)
  sqrd = (f.lambda/2)*vecnorm(x-p)^2
  gamlam = f.lambda*gamma
  return 1/(1+gamlam)*x + gamlam/(1+gamlam)*p, sqrd/(1+gamlam)^2
end

fun_name(f::SqrDistL2) = "squared Euclidean distance from a convex set"
fun_type(f::SqrDistL2) = "C^n → R"
fun_expr(f::SqrDistL2) = "x ↦ (λ/2) inf { ||x-y||^2 : y ∈ S} "
fun_params(f::SqrDistL2) = string("λ = $(f.lambda), S = ", typeof(f.ind))
