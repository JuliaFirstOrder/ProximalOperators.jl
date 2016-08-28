# L0 pseudo-norm (times a constant)

"""
  NormL0(λ::Float64=1.0)

Returns the function `g(x) = λ*countnz(x)`, for a nonnegative parameter `λ ⩾ 0`.
"""

immutable NormL0 <: ProximableFunction
  lambda::Float64
  NormL0(lambda::Float64=1.0) =
    lambda < 0 ? error("parameter λ must be nonnegative") : new(lambda)
end

@compat function (f::NormL0)(x::RealOrComplexArray)
  return f.lambda*countnz(x)
end

function prox(f::NormL0, x::RealOrComplexArray, gamma::Float64=1.0)
  over = abs(x) .> sqrt(2*gamma*f.lambda);
  y = x.*over;
  return y, f.lambda*countnz(y)
end

fun_name(f::NormL0) = "weighted L0 pseudo-norm"
fun_type(f::NormL0) = "C^n → R"
fun_expr(f::NormL0) = "x ↦ λ countnz(x)"
fun_params(f::NormL0) = "λ = $(f.lambda)"
