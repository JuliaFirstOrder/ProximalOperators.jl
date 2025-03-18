# L-infinity norm

export NormLinf

"""
    NormLinf(λ=1)

Return the ``L_∞`` norm
```math
f(x) = λ⋅\\max\\{|x_1|, …, |x_n|\\},
```
for a nonnegative parameter `λ`.
"""
NormLinf(lambda::T=1) where T = Conjugate(IndBallL1(lambda))

is_locally_smooth(f::Type{<:Conjugate{<:IndBallL1}}) = true

(f::Conjugate{<:IndBallL1})(x) = (f.f.r) * norm(x, Inf)

function gradient!(y, f::Conjugate{<:IndBallL1}, x)
    absxi, i = findmax(abs.(x)) # Largest absolute value
    y .= 0
    y[i] = f.f.r * sign(x[i])
    return f.f.r * absxi
end
