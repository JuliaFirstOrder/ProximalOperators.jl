# Separable sum

immutable SeparableSum{P<:ProximableFunction, I<:AbstractArray} <: ProximableFunction
	prox_col::Vector{P}
	ind_col::Array{I}

	function (::Type{SeparableSum}){P,I}(a::Vector{P},b::Array{I})
		new{P,I}(a, b)
	end
end


function prox!{T <: RealOrComplex, R <: Real}(f::SeparableSum, x::AbstractArray{T}, 
					      y::AbstractArray{T}, gamma::R=1.0)
  v = 0.0
  for i in eachindex(f.prox_col)
	  z = y[f.ind_col[i]]
	  g = prox!(f.prox_col[i],x[f.ind_col[i]],z,gamma)
	  y[f.ind_col[i]] = z
	  v += g
  end
  return v

end
