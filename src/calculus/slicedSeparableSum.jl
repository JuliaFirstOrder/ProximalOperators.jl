# Separable sum, using slices of an array as variables

immutable SlicedSeparableSum{S <: AbstractArray, T <: AbstractArray} <: ProximableFunction
	fs::S
	is::T
	dim::Integer
	function SlicedSeparableSum(a::S, b::T, dim::Integer)
		if size(a) != size(b)
			error("size(fs) must coincide with size(is)")
		else
			new(a, b, dim)
		end
	end
end

SlicedSeparableSum{S <: AbstractArray, T <: AbstractArray}(a::S, b::T, dim::Integer=1) =
	SlicedSeparableSum{S, T}(a, b, dim)

function SlicedSeparableSum{T <: Tuple}(ps::AbstractArray{T}, dim::Integer=1)
	a = Array{ProximableFunction}(length(ps))
	b = Array{AbstractArray}(length(ps))
	for i in eachindex(ps)
		a[i] = ps[i][1]
		b[i] = ps[i][2]
	end
	SlicedSeparableSum{typeof(a), typeof(b)}(a, b, dim)
end

function SlicedSeparableSum{P <: Pair}(p::AbstractArray{P}, dim::Integer = 1)
	a = Vector{ProximableFunction}(length(p))
	b = Vector{AbstractArray}(length(p))
	for i in eachindex(p)
		a[i] = p[i].first
		b[i] = p[i].second
	end
	return SlicedSeparableSum(a, b, dim)
end

function (f::SlicedSeparableSum{S}){S <: AbstractArray, T <: AbstractArray}(x::T)
	sum = 0.0
  for k in eachindex(f.fs)
		sum += f.fs[k](x[is[k]])
	end
	return sum
end

function prox!{T <: RealOrComplex}(y::AbstractArray{T}, f::SlicedSeparableSum, x::AbstractArray{T}, gamma::Real=1.0)
  v = 0.0
  nd = ndims(x)

  for i in eachindex(f.fs)
	  z = slicedim(y, f.dim, f.is[i])
	  g = prox!(z, f.fs[i], slicedim(x, f.dim, f.is[i]), gamma)
	  y[[ n==f.dim ? f.is[i] : indices(y, n) for n in 1:nd ]...] = z
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
