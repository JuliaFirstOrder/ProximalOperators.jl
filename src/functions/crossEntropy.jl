# Cross Entropy loss function

export CrossEntropy

"""
**Cross Entropy loss**

    CrossEntropy(b)

Returns the function
```math
f(x) = -\\frac{1}{N} \\sum_{i = 1}^{N} b_i \\log (x_i)+(1-b_i) \\log (1-x_i),
```
where `b` is an array such that `0 ≤ b ≤ 1` component-wise.
"""
struct CrossEntropy{R <: Real, T <: AbstractArray{R}} <: ProximableFunction
    b::T
    function CrossEntropy{R, T}(b::T) where {R <: Real, T <: AbstractArray{R}}
        if !(all(0 .<= b .<= 1. ))
            error("b must be 0 ≤ b ≤ 1 ")
        else
            new(b)
        end
    end
end

is_convex(f::CrossEntropy) = true
is_smooth(f::CrossEntropy) = true

CrossEntropy(b::T) where {R <: Real, T <: AbstractArray{R}} = CrossEntropy{R, T}(b)

function (f::CrossEntropy{R})(x::AbstractArray{R}) where {R <: Real}
    sum = R(0)
    for i in eachindex(f.b)
        sum += f.b[i]*log(x[i])+(1-f.b[i])*log(1-x[i])
    end
    return -sum/length(f.b)
end

function (f::CrossEntropy{B})(x::AbstractArray{R}) where {B <: Bool, R <: Real}
    sum = R(0)
    for i in eachindex(f.b)
        sum += f.b[i] ? log(x[i]) : (1-f.b[i])*log(1-x[i])
    end
    return -sum/length(f.b)
end

function gradient!(y::AbstractArray{R}, f::CrossEntropy{R}, x::AbstractArray{R}) where {R <: Real}
    sum = R(0)
    for i in eachindex(x)
        y[i] = 1/length(f.b)*( - f.b[i]/x[i] + (1-f.b[i])/(1-x[i]) )
        sum += f.b[i]*log(x[i])+(1-f.b[i])*log(1-x[i])
    end
    return -1/length(f.b)*sum
end

function gradient!(y::AbstractArray{R}, f::CrossEntropy{B}, x::AbstractArray{R}) where {R <: Real, B <: Bool}
    sum = R(0)
    for i in eachindex(x)
        y[i] = f.b[i] ? - 1/x[i] : 1/(1-x[i])
        y[i] *= 1/length(f.b)
        sum += f.b[i] ? log(x[i]) : log(1-x[i])
    end
    return -1/length(f.b)*sum
end

function prox!(y::AbstractArray{R}, f::CrossEntropy, x::AbstractArray{R}, gamma::R=R(1)) where {R}
    # TODO: fill-in here
    error("not implemented")
end

fun_name(f::CrossEntropy) = "cross entropy loss"
fun_dom(f::CrossEntropy) = "AbstractArray{Real}"
fun_expr(f::CrossEntropy) = "x ↦ 1/N sum( b*log(x)+(1-b)*log(1-x), i=1,...,N )"
fun_params(f::CrossEntropy) = "b = $(typeof(f.b))"
