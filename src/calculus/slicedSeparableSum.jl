# Separable sum

immutable SlicedSeparableSum{P<:ProximableFunction, I<:AbstractArray} <: ProximableFunction
	prox_col::Vector{P}
	ind_col::Array{I}
	dim::Int64

	function (::Type{SlicedSeparableSum}){P,I}(a::Vector{P},b::Array{I}, dim::Int64 = 1)
		if length(a)!= length(b)
			error("length(prox_col) must conicide with length(ind_col)")
		else
			new{P,I}(a, b, dim)
		end
	end
end

function SlicedSeparableSum(p::Pair...; dim::Int64 = 1)

	a = Vector{ProximableFunction}(length(p))
	b = Vector{AbstractArray}(length(p))

	for i in eachindex(p)
		a[i] = p[i].first
		b[i]  = p[i].second
	end
	return SlicedSeparableSum(a, b, dim)
	
end

function prox!{T <: RealOrComplex, R <: Real}(f::SlicedSeparableSum, x::AbstractArray{T}, 
					      y::AbstractArray{T}, gamma::R=1.0)
  v = 0.0
  nd = ndims(x)

  for i in eachindex(f.prox_col)
	  z = slicedim(y,f.dim,f.ind_col[i])
	  g = prox!(f.prox_col[i],slicedim(x,f.dim,f.ind_col[i]),z,gamma)
	  y[[ n==f.dim ? f.ind_col[i] : indices(y,n) for n in 1:nd ]...] = z
	  v += g
  end
  return v

end

function Base.show(io::IO, f::SlicedSeparableSum)  

	println(io,"Sliced Separable Sum: \n")
	if length(f.prox_col)<=4
		for p in f.prox_col
			show(p)
			println(io,"\n")
		end
	else
		show(f.prox_col[1])
		println(io,"\n .... \n")
		show(f.prox_col[end])
	end
end
