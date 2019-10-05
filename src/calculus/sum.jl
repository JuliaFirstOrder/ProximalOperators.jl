export Sum

"""
**Sum of functions**

    Sum(f_1, ..., f_k)

Given functions `f_1` to `f_k`, returns their sum

```math
g(x) = \\sum_{i=1}^k f_i(x).
```
"""
struct Sum{T <: Tuple} <: ProximableFunction fs::T end

Sum(fs::Vararg{ProximableFunction}) = Sum((fs...,))

# note: is_prox_accurate false because prox in general doesn't exist?
is_prox_accurate(f::Sum) = false
is_convex(f::Sum) = all(is_convex.(f.fs))
is_set(f::Sum) = all(is_set.(f.fs))
is_cone(f::Sum) = all(is_cone.(f.fs))
is_affine(f::Sum) = all(is_affine.(f.fs))
is_smooth(f::Sum) = all(is_smooth.(f.fs))
is_quadratic(f::Sum) = all(is_quadratic.(f.fs))
is_generalized_quadratic(f::Sum) = all(is_generalized_quadratic.(f.fs))
is_strongly_convex(f::Sum) = all(is_convex.(f.fs)) && any(is_strongly_convex.(f.fs))

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

fun_name(f::Sum) = "sum"
fun_dom(f::Sum) = fun_dom(f.fs[1]) # check to make sure same??
fun_expr(f::Sum) = "x ↦ f₁(x) + … + fₖ(x)"
fun_params(f::Sum) = "n/a"
