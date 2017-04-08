# Separable sum, using slices of an array as variables

immutable SlicedSeparableSum{S <: AbstractArray, T <: AbstractArray} <: ProximableFunction
	fs::S
	idxs::T
	function SlicedSeparableSum(fs, idxs)
		if size(fs) != size(idxs)
			error("size(fs) must coincide with size(idxs)")
		else
			verify_idx = true
			for i in eachindex(idxs)
				verify_idx *=
				all([typeof(t) <: AbstractArray{Int,1} ||
	                             typeof(t) <: Colon for t in idxs[i]])
			end
			verify_idx ? new(fs, idxs) :error("invalid index")
		end
	end
end

is_separable(f::SlicedSeparableSum) = all(is_separable.(f.fs))
is_prox_accurate(f::SlicedSeparableSum) = all(is_prox_accurate.(f.fs))
is_convex(f::SlicedSeparableSum) = all(is_convex.(f.fs))
is_set(f::SlicedSeparableSum) = all(is_set.(f.fs))
is_cone(f::SlicedSeparableSum) = all(is_cone.(f.fs))

SlicedSeparableSum{S <: AbstractArray, T <: AbstractArray}(a::S, b::T) =
SlicedSeparableSum{S, T}(a, b)

function (f::SlicedSeparableSum{S}){S <: AbstractArray, T <: AbstractArray}(x::T)
	sum = 0.0
	for k in eachindex(f.fs)
		sum += f.fs[k](x[f.idx[k]])
	end
	return sum
end

function prox!{T <: RealOrComplex}(y::AbstractArray{T},
				   f::SlicedSeparableSum, x::AbstractArray{T}, gamma::Real=1.0)
  v = 0.0
  for k in eachindex(f.fs)
	  g = prox!(view(y,f.idxs[k]...), f.fs[k], view(x,f.idxs[k]...), gamma)
	  v += g
  end
  return v

end

fun_name(f::SlicedSeparableSum) = "sliced separable sum"
function fun_dom(f::SlicedSeparableSum)
	# s = ""
	# for k in eachindex(f.fs)
	# 	s = string(s, fun_dom(f.fs[k]), " × ")
	# end
	# return s
	return "n/a" # for now
end
fun_expr(f::SlicedSeparableSum) = "hard to explain"
function fun_params(f::SlicedSeparableSum)
	# s = "r = $(size(f.fs)), "
	# for k in eachindex(f.fs)
	# 	s = string(s, "f[$(k)] = $(typeof(f.fs[k])), ")
	# end
	# return s
	return "n/a" # for now
end
