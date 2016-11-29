# Separable sum

immutable SlicedSeparableSum{S <: Union{AbstractArray, Tuple}, T <: Union{AbstractArray, Tuple}} <: ProximableFunction
	fs::S
	is::T
	dim::Integer

	function SlicedSeparableSum(a::S, b::T, dim::Integer)
		if length(a) != length(b)
			error("length(fs) must coincide with length(is)")
		else
			new(a, b, dim)
		end
	end
end

SlicedSeparableSum{S <: Union{AbstractArray, Tuple}, T <: Union{AbstractArray, Tuple}}(a::S, b::T, dim::Integer=1) =
	SlicedSeparableSum{S, T}(a, b, dim)

function SlicedSeparableSum{T <: Union{AbstractArray, Tuple}}(ps::T, dim::Integer=1)
	a = Array{ProximableFunction}(length(ps))
	b = Array{AbstractArray}(length(ps))
	for i = 1:length(ps)
		a[i] = ps[i][1]
		b[i] = ps[i][2]
	end
	SlicedSeparableSum{typeof(a),typeof(b)}(a, b, dim)
end

function prox!{T <: RealOrComplex}(f::SlicedSeparableSum, x::AbstractArray{T}, y::AbstractArray{T}, gamma::Real=1.0)
  v = 0.0
  nd = ndims(x)

  for i = 1:length(f.fs)
	  z = slicedim(y,f.dim,f.is[i])
	  g = prox!(f.fs[i],slicedim(x,f.dim,f.is[i]),z,gamma)
	  y[[ n==f.dim ? f.is[i] : indices(y,n) for n in 1:nd ]...] = z
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
