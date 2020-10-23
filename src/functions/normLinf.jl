# L-infinity norm

export NormLinf

"""
**``L_∞`` norm**

    NormLinf(λ=1)

Returns the function
```math
f(x) = λ⋅\\max\\{|x_1|, …, |x_n|\\},
```
for a nonnegative parameter `λ`.
"""
NormLinf(lambda::T=1) where {T <: Real} = Conjugate(IndBallL1(lambda))

function (f::Conjugate{IndBallL1{T}})(x::AbstractArray{C}) where {T, R <: Real, C <: RealOrComplex{R}}
    return (f.f.r) * norm(x, Inf)
end

function gradient!(y::AbstractArray{C}, f::Conjugate{IndBallL1{T}}, x::AbstractArray{C}) where {
    T, R <: Real, C <: RealOrComplex{R}
}
    absxi, i = findmax(abs.(x)) # Largest absolute value
    y .= 0
    y[i] = f.f.r * sign(x[i])
    return f.f.r * absxi
end

fun_name(f::Postcompose{Conjugate{IndBallL1{R}}, R}) where {R <: Real} = "weighted L-infinity norm"
fun_expr(f::Postcompose{Conjugate{IndBallL1{R}}, R}) where {R <: Real} = "x ↦ λ||x||_∞ = λ⋅max(abs(x))"
fun_params(f::Postcompose{Conjugate{IndBallL1{R}}, R}) where {R <: Real} = "λ = $(f.a*(f.f.f.r))"
