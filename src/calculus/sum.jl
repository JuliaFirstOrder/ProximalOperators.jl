export Sum

"""
    Sum(f_1, ..., f_k)

Given functions `f_1` to `f_k`, return their sum

```math
g(x) = \\sum_{i=1}^k f_i(x).
```
"""
struct Sum{T <: Tuple} fs::T end

Sum(fs::Vararg) = Sum((fs...,))

# note: is_prox_accurate false because prox in general doesn't exist?
is_prox_accurate(::Type{<:Sum{T}}) where T = false
is_convex(::Type{<:Sum{T}}) where T = all(is_convex.(T.parameters))
is_set(::Type{<:Sum{T}}) where T = all(is_set.(T.parameters))
is_cone(::Type{<:Sum{T}}) where T = all(is_cone.(T.parameters))
is_affine(::Type{<:Sum{T}}) where T = all(is_affine.(T.parameters))
is_smooth(::Type{<:Sum{T}}) where T = all(is_smooth.(T.parameters))
is_generalized_quadratic(::Type{<:Sum{T}}) where T = all(is_generalized_quadratic.(T.parameters))
is_strongly_convex(::Type{<:Sum{T}}) where T = all(is_convex.(T.parameters)) && any(is_strongly_convex.(T.parameters))

function (sumobj::Sum)(x::AbstractArray{T}) where {R <: Real, T <: Union{R, Complex{R}}}
    sum = R(0)
    for f in sumobj.fs
        sum += f(x)
    end
    sum
end

function gradient!(grad::AbstractArray{T}, sumobj::Sum, x::AbstractArray{T}) where {R <: Real, T <: Union{R, Complex{R}}}
    # gradient of sum is sum of gradients
    val = R(0)
    # to keep track of this sum, i may not be able to
    # avoid allocating an array
    grad .= T(0)
    temp = similar(grad)
    for f in sumobj.fs
        val += gradient!(temp, f, x)
        grad .+= temp
    end
    return val
end
