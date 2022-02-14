# L0 pseudo-norm (times a constant)

export NormL0

"""
    NormL0(λ=1)

Return the ``L_0`` pseudo-norm function
```math
f(x) = λ\\cdot\\mathrm{nnz}(x)
```
for a nonnegative parameter `λ`.
"""
struct NormL0{R}
    lambda::R
    function NormL0{R}(lambda::R) where R
        if lambda < 0
            error("parameter λ must be nonnegative")
        else
            new(lambda)
        end
    end
end

NormL0(lambda::R=1) where R = NormL0{R}(lambda)

(f::NormL0)(x) = f.lambda * real(eltype(x))(count(!iszero, x))

function prox!(y, f::NormL0, x, gamma)
    countnzy = real(eltype(x))(0)
    gl = gamma * f.lambda
    for i in eachindex(x)
        over = abs(x[i]) > sqrt(2 * gl)
        y[i] = over * x[i]
        countnzy += over
    end
    return f.lambda * countnzy
end

function prox_naive(f::NormL0, x, gamma)
    over = abs.(x) .> sqrt(2 * gamma * f.lambda)
    y = x.*over
    return y, f.lambda * real(eltype(x))(count(!iszero, y))
end
