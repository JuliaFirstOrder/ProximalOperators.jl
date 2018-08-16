# Generalized length, dot product, norm, similar and deepcopy for tuples

# deepsimilar(x::Tuple) = deepsimilar.(x)
#
# deepsimilar(x::AbstractArray) = similar(x)

import Base: isapprox, similar

isapprox(x::Tuple, y::Tuple) = all(isapprox.(x, y))

similar(x::Tuple) = similar.(x)

#
# deepcopy!(y::T, x::T) where {T <: Tuple} = deepcopy!.(y, x)
#
# deepcopy!(y::AbstractArray{R}, x::AbstractArray{R}) where {R <: Number} = copy!(y, x)
#
# deeplength(x::Tuple) = sum(deeplength.(x))
#
# deeplength(x::AbstractArray) = length(x)
#
# deepsize(x::Tuple) = map(deepsize, x)
#
# deepsize(x::AbstractArray) = size(x)
#
# deepvecdot(x::T, y::T) where {T <: Tuple} = sum(deepvecdot.(x,y))
#
# deepvecdot(x::AbstractArray{R}, y::AbstractArray{R}) where {R <: Number} = vecdot(x, y)
#
# deepvecnorm(x::Tuple) = sqrt(deepvecdot(x, x))
#
# deepvecnorm(x::AbstractArray{R}) where {R <: Number} = vecnorm(x)
#
# deepmaxabs(x::Tuple) = maximum(deepmaxabs.(x))
#
# deepmaxabs(x::AbstractArray{R}) where {R <: Number} = maximum(abs, x)
#
# deepzeros(t::Tuple, s::Tuple) = deepzeros.(t, s)
#
# deepzeros(t::Type, n::NTuple{N, Integer} where {N}) = zeros(t, n)
#
# deepaxpy!(z::T, x::T, alpha::R, y::T) where {T <: Tuple, R <: Real} = deepaxpy!.(z, x, alpha, y)
#
# deepaxpy!(z::AbstractArray{T}, x::AbstractArray{T}, alpha::R, y::AbstractArray{T}) where {T <: Number, R <: Real} = (z .= x .+ alpha.*y)
