# Generalized length, dot product, norm, similar and deepcopy for nested Array objects

function deepsimilar(x::AbstractArray)
	y = similar(x)
	for k in eachindex(x)
		y[k] = similar(x[k])
	end
	return y
end

deepsimilar{T <: Number}(x::AbstractArray{T}) = similar(x)

function deepcopy!(y::AbstractArray, x::AbstractArray)
	for k in eachindex(x)
		copy!(y[k],x[k])
	end
end

deepcopy!{T <: Number}(y::AbstractArray{T}, x::AbstractArray{T}) = copy!(y,x)

function deeplength(x::AbstractArray)
  len = 0
	for k in eachindex(x)
		len += deeplength(x[k])
	end
	return len
end

deeplength{T <: Number}(x::AbstractArray{T}) = length(x)

function deepvecdot(x::AbstractArray, y::AbstractArray)
	out = 0.0
	for k in eachindex(x)
		out += deepvecdot(x[k], y[k])
	end
	return out
end

deepvecdot{T <: Number}(x::AbstractArray{T}, y::AbstractArray{T}) = vecdot(x, y)

deepvecnorm(x::AbstractArray) = sqrt(deepvecdot(x, x))

deepvecnorm{T <: Number}(x::AbstractArray{T}) = vecnorm(x)

function deepmaxabs(x::AbstractArray)
	out = 0.0
	for k in eachindex(x)
		out = max(out, deepmaxabs(x[k]))
	end
	return out
end

deepmaxabs{T <: Number}(x::AbstractArray{T}) = maxabs(x)
