
immutable SeparableSum{P<: ProximableFunction} <: ProximableFunction
	prox_col::Vector{P}
end

function SeparableSum(pr::Vector{DataType}, lambdas::Vector)
	prox_col = Vector{ProximableFunction}(length(lambdas)) 
	if length(pr)!=length(lambdas)
		error("lambda vector and proximal operator vecor must have the same lenght")
	else
		for i in eachindex(pr)
			prox_col[i] = pr[i](lambdas[i])
		end
		return SeparableSum(prox_col)
	end

end

function prox!{R <: Real}(f::SeparableSum, x::Vector, 
			  y::Vector, gamma::R=1.0)
  v = 0.0
  for i in eachindex(f.prox_col)
	  g = prox!(f.prox_col[i],x[i],y[i],gamma)
	  v += g
  end
  return v

end

function prox(f::SeparableSum, x::Vector, gamma::Real=1.0)
	y = Vector(length(x))
	for i in eachindex(x)
		y[i] = similar(x[i])
	end
	fy = prox!(f, x, y, gamma)
	return y, fy
end

function Base.show(io::IO, f::SeparableSum)  

	println(io,"Separable Sum: \n")
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
