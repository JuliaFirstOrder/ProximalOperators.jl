# Extension of some common Array operations to Tuple objects

import Base: isapprox, similar, zero

isapprox(x::Tuple, y::Tuple, args...; kwargs...) = all(isapprox.(x, y, args...; kwargs...))

similar(x::Tuple) = similar.(x)

zero(x::Tuple) = zero.(x)
