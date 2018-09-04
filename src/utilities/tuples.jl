# Extension of some common Array operations to Tuple objects

import Base: isapprox, similar

isapprox(x::Tuple, y::Tuple) = all(isapprox.(x, y))

similar(x::Tuple) = similar.(x)
