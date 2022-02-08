# Separable sum, using tuples of arrays as variables

export SeparableSum

"""
    SeparableSum(f_1, ..., f_k)

Given functions `f_1` to `f_k`, return their separable sum, that is
```math
g(x_1, ..., x_k) = \\sum_{i=1}^k f_i(x_i).
```
The object `g` constructed in this way can be evaluated at `Tuple`s of length
`k`. Likewise, the `prox` and `prox!` methods for `g` operate with
(input and output) `Tuple`s of length `k`.

Example:

    f = SeparableSum(NormL1(), NuclearNorm()); # separable sum of two functions
    x = randn(10); # some random vector
    Y = randn(20, 30); # some random matrix
    f_xY = f((x, Y)); # evaluates f at (x, Y)
    (u, V), f_uV = prox(f, (x, Y), 1.3); # computes prox at (x, Y)
"""
struct SeparableSum{T <: Tuple}
    fs::T
end

SeparableSum(fs::Vararg) = SeparableSum((fs...,))

is_prox_accurate(::Type{<:SeparableSum{T}}) where T = all(is_prox_accurate.(T.parameters))
is_convex(::Type{<:SeparableSum{T}}) where T = all(is_convex.(T.parameters))
is_set(::Type{<:SeparableSum{T}}) where T = all(is_set.(T.parameters))
is_singleton(::Type{<:SeparableSum{T}}) where T = all(is_singleton.(T.parameters))
is_cone(::Type{<:SeparableSum{T}}) where T = all(is_cone.(T.parameters))
is_affine(::Type{<:SeparableSum{T}}) where T = all(is_affine.(T.parameters))
is_smooth(::Type{<:SeparableSum{T}}) where T = all(is_smooth.(T.parameters))
is_generalized_quadratic(::Type{<:SeparableSum{T}}) where T = all(is_generalized_quadratic.(T.parameters))
is_strongly_convex(::Type{<:SeparableSum{T}}) where T = all(is_strongly_convex.(T.parameters))

function (f::SeparableSum)(x::TupleOfArrays{R}) where R <: Real
    sum = R(0)
    for k in eachindex(x)
        sum += f.fs[k](x[k])
    end
    return sum
end

function prox!(ys::TupleOfArrays{R}, fs::Tuple, xs::TupleOfArrays{R}, gamma::Number) where R <: Real
    sum = R(0)
    for k in eachindex(xs)
        sum += prox!(ys[k], fs[k], xs[k], gamma)
    end
    return sum
end

function prox!(ys::TupleOfArrays{R}, fs::Tuple, xs::TupleOfArrays{R}, gamma::Tuple) where R <: Real
    sum = R(0)
    for k in eachindex(xs)
        sum += prox!(ys[k], fs[k], xs[k], gamma[k])
    end
    return sum
end

prox!(ys::TupleOfArrays{R}, f::SeparableSum, xs::TupleOfArrays{R}, gamma) where R <: Real = prox!(ys, f.fs, xs, gamma)

function gradient!(grad::TupleOfArrays{R}, fs::Tuple, x::TupleOfArrays{R}) where R <: Real
    val = R(0)
    for k in eachindex(fs)
        val += gradient!(grad[k], fs[k], x[k])
    end
    return val
end

gradient!(grad::TupleOfArrays, f::SeparableSum, x::TupleOfArrays) = gradient!(grad, f.fs, x)

fun_name(f::SeparableSum) = "separable sum"
fun_dom(f::SeparableSum) = "n/a"
fun_expr(f::SeparableSum) = "(x₁, …, xₖ) ↦ f₁(x₁) + … + fₖ(xₖ)"
fun_params(f::SeparableSum) = "n/a"

function prox_naive(f::SeparableSum, xs::TupleOfArrays{R}, gamma) where R <: Real
    fys = R(0)
    ys = []
    for k in eachindex(xs)
        y, fy = prox_naive(f.fs[k], xs[k], typeof(gamma) <: Number ? gamma : gamma[k])
        fys += fy
        append!(ys, [y])
    end
    return Tuple(ys), fys
end
