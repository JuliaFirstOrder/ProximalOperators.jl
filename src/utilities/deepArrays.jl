# Generalized length, dot product, norm, similar and deepcopy for tuples

deepsimilar(x::Tuple) = deepsimilar.(x)

deepsimilar(x::AbstractArray) = similar(x)

deepcopy!{T <: Tuple}(y::T, x::T) = deepcopy!.(y, x)

deepcopy!{R <: Number}(y::AbstractArray{R}, x::AbstractArray{R}) = copy!(y, x)

deeplength(x::Tuple) = sum(deeplength.(x))

deeplength(x::AbstractArray) = length(x)

deepvecdot{T <: Tuple}(x::T, y::T) = sum(deepvecdot.(x,y))

deepvecdot{R <: Number}(x::AbstractArray{R}, y::AbstractArray{R}) = vecdot(x, y)

deepvecnorm(x::Tuple) = sqrt(deepvecdot(x, x))

deepvecnorm{R <: Number}(x::AbstractArray{R}) = vecnorm(x)

deepmaxabs(x::Tuple) = maximum(deepmaxabs.(x))

deepmaxabs{R <: Number}(x::AbstractArray{R}) = maximum(abs, x)

deepzeros(t::Tuple, s::Tuple) = zeros.(t, s)

deepzeros(t::Type, s::AbstractArray) = zeros(t, s)
