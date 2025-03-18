export Sum

"""
    Sum(f_1, ..., f_k)

Given functions `f_1` to `f_k`, return their sum

```math
g(x) = \\sum_{i=1}^k f_i(x).
```
"""
struct Sum{T}
    fs::T
end

Sum(fs::Vararg) = Sum((fs...,))

component_types(::Type{Sum{T}}) where T = fieldtypes(T)

# note: is_proximable false because prox in general doesn't exist?
is_proximable(::Type{<:Sum}) = false
@generated is_convex(::Type{T}) where T <: Sum = return all(is_convex, component_types(T)) ? true : false
@generated is_set_indicator(::Type{T}) where T <: Sum = return all(is_set_indicator, component_types(T)) ? true : false
@generated is_singleton_indicator(::Type{T}) where T <: Sum = return all(is_singleton_indicator, component_types(T)) ? true : false
@generated is_cone_indicator(::Type{T}) where T <: Sum = return all(is_cone_indicator, component_types(T)) ? true : false
@generated is_affine_indicator(::Type{T}) where T <: Sum = return all(is_affine_indicator, component_types(T)) ? true : false
@generated is_smooth(::Type{T}) where T <: Sum = return all(is_smooth, component_types(T)) ? true : false
@generated is_locally_smooth(::Type{T}) where T <: Sum = return all(is_locally_smooth, component_types(T)) ? true : false
@generated is_generalized_quadratic(::Type{T}) where T <: Sum = return all(is_generalized_quadratic, component_types(T)) ? true : false
@generated is_strongly_convex(::Type{T}) where T <: Sum = return (all(is_convex, component_types(T)) && any(is_strongly_convex, component_types(T))) ? true : false

function (sumobj::Sum)(x)
    sum = real(eltype(x))(0)
    for f in sumobj.fs
        sum += f(x)
    end
    sum
end

function gradient!(grad, sumobj::Sum, x)
    # gradient of sum is sum of gradients
    val = real(eltype(x))(0)
    # to keep track of this sum, i may not be able to
    # avoid allocating an array
    grad .= eltype(x)(0)
    temp = similar(grad)
    for f in sumobj.fs
        val += gradient!(temp, f, x)
        grad .+= temp
    end
    return val
end
