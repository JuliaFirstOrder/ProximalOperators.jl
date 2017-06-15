# Separable sum, using tuples of arrays as variables

export SeparableSum

"""
**Separable sum of functions**

    SeparableSum(f₁,…,fₖ)

Given functions `f₁` to `fₖ`, returns their separable sum, that is
```math
g(x_1,…,x_k) = ∑_\{i=1\}^k f_i(x_i).
```
"""

immutable SeparableSum{T <: Tuple} <: ProximableFunction
	fs::T
end

SeparableSum(fs::Vararg{ProximableFunction}) = SeparableSum((fs...))

is_prox_accurate(f::SeparableSum) = all(is_prox_accurate.(f.fs))
is_convex(f::SeparableSum) = all(is_convex.(f.fs))
is_set(f::SeparableSum) = all(is_set.(f.fs))
is_singleton(f::SeparableSum) = all(is_singleton.(f.fs))
is_cone(f::SeparableSum) = all(is_cone.(f.fs))
is_affine(f::SeparableSum) = all(is_affine.(f.fs))
is_smooth(f::SeparableSum) = all(is_smooth.(f.fs))
is_quadratic(f::SeparableSum) = all(is_quadratic.(f.fs))
is_generalized_quadratic(f::SeparableSum) = all(is_generalized_quadratic.(f.fs))
is_strongly_convex(f::SeparableSum) = all(is_strongly_convex.(f.fs))

function (f::SeparableSum)(x::Tuple)
	sum = 0.0
  for k in eachindex(x)
	  sum += f.fs[k](x[k])
  end
  return sum
end

function prox!{T <: Tuple}(ys::T, fs::Tuple, xs::T, gamma::Real=1.0)
  sum = 0.0
  for k in eachindex(xs)
	  sum += prox!(ys[k], fs[k], xs[k], gamma)
  end
  return sum
end

function prox!{T <: Tuple}(ys::T, fs::Tuple, xs::T, gamma::Tuple)
  sum = 0.0
  for k in eachindex(xs)
	  sum += prox!(ys[k], fs[k], xs[k], gamma[k])
  end
  return sum
end

prox!{T <: Tuple}(ys::T, f::SeparableSum, xs::T, gamma::Union{Real, Tuple}=1.0) = prox!(ys, f.fs, xs, gamma)

function gradient!{T <: Tuple}(grad::T, fs::Tuple, x::T)
  val = 0.0
  for k in eachindex(fs)
    val += gradient!(grad[k], fs[k], x[k])
  end
  return val
end

gradient!{T <: Tuple}(grad::T, f::SeparableSum, x::T) = gradient!(grad, f.fs, x)

fun_name(f::SeparableSum) = "separable sum"
fun_dom(f::SeparableSum) = "n/a"
fun_expr(f::SeparableSum) = "(x₁, …, xₖ) ↦ f₁(x₁) + … + fₖ(xₖ)"
fun_params(f::SeparableSum) = "n/a"

function prox_naive(f::SeparableSum, xs::Tuple, gamma::Union{Real, Tuple}=1.0)
	fys = 0.0
	ys = [];
  for k in eachindex(xs)
	  y, fy = prox_naive(f.fs[k], xs[k], typeof(gamma) <: Real ? gamma : gamma[k])
		fys += fy;
		append!(ys, [y]);
  end
	return Tuple(ys), fys
end
