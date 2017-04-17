# Separable sum, using arrays of arrays as variables

immutable SeparableSum{S <: AbstractArray} <: ProximableFunction
	fs::S
end

SeparableSum(fs::Vararg{ProximableFunction}) = SeparableSum([fs...])

is_separable(f::SeparableSum) = all(is_separable.(f.fs))
is_prox_accurate(f::SeparableSum) = all(is_prox_accurate.(f.fs))
is_convex(f::SeparableSum) = all(is_convex.(f.fs))
is_set(f::SeparableSum) = all(is_set.(f.fs))
is_cone(f::SeparableSum) = all(is_cone.(f.fs))

function (f::SeparableSum{S}){S <: AbstractArray}(x::AbstractArray)
	sum = 0.0
  for k in eachindex(f.fs)
	  sum += f.fs[k](x[k])
  end
  return sum
end

function prox!(ys::AbstractArray, fs::AbstractArray, xs::AbstractArray, gamma::Real=1.0)
  sum = 0.0
  for k in eachindex(fs)
	  sum += prox!(ys[k], fs[k], xs[k], gamma)
  end
  return sum
end

function prox(fs::AbstractArray, xs::AbstractArray, gamma::Real=1.0)
	ys = map(similar, xs)
	fsy = prox!(ys, fs, xs, gamma)
	return ys, fsy
end

prox!(ys::AbstractArray, f::SeparableSum, xs::AbstractArray, gamma::Real=1.0) =
	prox!(ys, f.fs, xs, gamma)

prox(f::SeparableSum, xs::AbstractArray, gamma::Real=1.0) =
	prox(f.fs, xs, gamma)

fun_name(f::SeparableSum) = "separable sum"
function fun_dom(f::SeparableSum)
	# s = ""
	# for k in eachindex(f.fs)
	# 	s = string(s, fun_dom(f.fs[k]), " × ")
	# end
	# return s
	return "n/a" # for now
end
fun_expr(f::SeparableSum) = "(x₁,…,xᵣ) ↦ f[1](x₁) + … + f[r](xᵣ)"
function fun_params(f::SeparableSum)
	# s = "r = $(size(f.fs)), "
	# for k in eachindex(f.fs)
	# 	s = string(s, "f[$(k)] = $(typeof(f.fs[k])), ")
	# end
	# return s
	return "n/a" # for now
end

function prox_naive(f::SeparableSum, xs::AbstractArray, gamma::Real=1.0)
	fys = 0.0
	ys = [];
  for k in eachindex(f.fs)
	  y, fy = prox_naive(f.fs[k], xs[k], gamma)
		fys += fy;
		append!(ys, [y]);
  end
	return ys, fys
end
