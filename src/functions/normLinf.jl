# L-infinity norm

export NormLinf

"""
**``L_∞`` norm**

    NormLinf(λ=1.0)

Returns the function
```math
f(x) = λ⋅\\max\\{|x_1|, …, |x_n|\\},
```
for a nonnegative parameter `λ`.
"""
NormLinf(lambda::R=1.0) where {R <: Real} = Conjugate(IndBallL1(lambda))

function (f::Conjugate{IndBallL1{R}})(x::AbstractArray{S}) where {R <: Real, S <: RealOrComplex}
    return (f.f.r)*norm(x, Inf)
end

function gradient!(y::AbstractArray{T}, f::Conjugate{IndBallL1{R}}, x::AbstractArray{T}) where {T <: RealOrComplex, R <: Real}
    absxi, i = findmax(abs.(x)) # Largest absolute value
    y .= 0
    y[i] = f.f.r*sign(x[i])
    return f.f.r*absxi
end

fun_name(f::Postcompose{Conjugate{IndBallL1{R}}, R}) where {R <: Real} = "weighted L-infinity norm"
fun_expr(f::Postcompose{Conjugate{IndBallL1{R}}, R}) where {R <: Real} = "x ↦ λ||x||_∞ = λ⋅max(abs(x))"
fun_params(f::Postcompose{Conjugate{IndBallL1{R}}, R}) where {R <: Real} = "λ = $(f.a*(f.f.f.r))"
