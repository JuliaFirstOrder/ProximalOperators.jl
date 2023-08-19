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
struct SeparableSum{T}
    fs::T
end

SeparableSum(fs::Vararg) = SeparableSum((fs...,))

component_types(::Type{SeparableSum{T}}) where T = fieldtypes(T)

@generated is_prox_accurate(::Type{T}) where T <: SeparableSum = return all(is_prox_accurate, component_types(T)) ? :(true) : :(false)
@generated is_convex(::Type{T}) where T <: SeparableSum = return all(is_convex, component_types(T)) ? :(true) : :(false)
@generated is_set(::Type{T}) where T <: SeparableSum = return all(is_set, component_types(T)) ? :(true) : :(false)
@generated is_singleton(::Type{T}) where T <: SeparableSum = return all(is_singleton, component_types(T)) ? :(true) : :(false)
@generated is_cone(::Type{T}) where T <: SeparableSum = return all(is_cone, component_types(T)) ? :(true) : :(false)
@generated is_affine(::Type{T}) where T <: SeparableSum = return all(is_affine, component_types(T)) ? :(true) : :(false)
@generated is_smooth(::Type{T}) where T <: SeparableSum = return all(is_smooth, component_types(T)) ? :(true) : :(false)
@generated is_generalized_quadratic(::Type{T}) where T <: SeparableSum = return all(is_generalized_quadratic, component_types(T)) ? :(true) : :(false)
@generated is_strongly_convex(::Type{T}) where T <: SeparableSum = return all(is_strongly_convex, component_types(T)) ? :(true) : :(false)

(g::SeparableSum)(xs::Tuple) = sum(f(x) for (f, x) in zip(g.fs, xs))

prox!(ys::Tuple, g::SeparableSum, xs::Tuple, gamma::Number) = sum(prox!(y, f, x, gamma) for (y, f, x) in zip(ys, g.fs, xs))

prox!(ys::Tuple, g::SeparableSum, xs::Tuple, gammas::Tuple) = sum(prox!(y, f, x, gamma) for (y, f, x, gamma) in zip(ys, g.fs, xs, gammas))

function prox(g::SeparableSum, xs::Tuple, gamma=1)
    ys = similar.(xs)
    fys = prox!(ys, g, xs, gamma)
    return ys, fys
end

gradient!(grads::Tuple, g::SeparableSum, xs::Tuple) = sum(gradient!(grad, f, x) for (grad, f, x) in zip(grads, g.fs, xs))

function gradient(g::SeparableSum, xs::Tuple)
    ys = similar.(xs)
    fxs = gradient!(ys, g, xs)
    return ys, fxs
end

function prox_naive(f::SeparableSum, xs::Tuple, gamma)
    fys = real(eltype(xs[1]))(0)
    ys = []
    for k in eachindex(xs)
        y, fy = prox_naive(f.fs[k], xs[k], typeof(gamma) <: Number ? gamma : gamma[k])
        fys += fy
        append!(ys, [y])
    end
    return Tuple(ys), fys
end
