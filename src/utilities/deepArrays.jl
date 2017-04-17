# Generalized length, dot product, norm, similar and deepcopy for nested Array objects

function deepsimilar(x::AbstractArray)
	y = similar(x)
	for k in eachindex(x)
    if length(x[k]) > 1
      y[k] = deepsimilar(x[k])
    else
      y[k] = 0.0
    end
	end
	return y
end
deepsimilar{T <: Number}(x::AbstractArray{T}) = similar(x)

function deepcopy!(y::AbstractArray, x::AbstractArray)
	for k in eachindex(x)
    if length(x[k]) > 1
      deepcopy!(y[k],x[k])
    else
      y[k] = x[k]
    end
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
deeplength{T <: Number}(x::T) = 1

function deepvecdot(x::AbstractArray, y::AbstractArray)
	out = 0.0
	for k in eachindex(x)
		out += deepvecdot(x[k], y[k])
	end
	return out
end

deepvecdot{T <: Number}(x::AbstractArray{T}, y::AbstractArray{T}) = vecdot(x, y)
deepvecdot{T <: Number}(x::T, y::T) = x*y

deepvecnorm(x::AbstractArray) = sqrt(deepvecdot(x, x))
deepvecnorm{T <: Number}(x::AbstractArray{T}) = vecnorm(x)
deepvecnorm{T <: Number}(x::T) = abs(x)

function deepmaxabs(x::AbstractArray)
	out = 0.0
	for k in eachindex(x)
		out = max(out, deepmaxabs(x[k]))
	end
	return out
end

deepmaxabs{T <: Number}(x::AbstractArray{T}) = maxabs(x)
deepmaxabs{T <: Number}(x::T) = abs(x)
