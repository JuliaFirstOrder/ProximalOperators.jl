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

is_prox_accurate(::Type{<:SeparableSum{T}}) where T = all(is_prox_accurate.(T.parameters))
is_convex(::Type{<:SeparableSum{T}}) where T = all(is_convex.(T.parameters))
is_set(::Type{<:SeparableSum{T}}) where T = all(is_set.(T.parameters))
is_singleton(::Type{<:SeparableSum{T}}) where T = all(is_singleton.(T.parameters))
is_cone(::Type{<:SeparableSum{T}}) where T = all(is_cone.(T.parameters))
is_affine(::Type{<:SeparableSum{T}}) where T = all(is_affine.(T.parameters))
is_smooth(::Type{<:SeparableSum{T}}) where T = all(is_smooth.(T.parameters))
is_generalized_quadratic(::Type{<:SeparableSum{T}}) where T = all(is_generalized_quadratic.(T.parameters))
is_strongly_convex(::Type{<:SeparableSum{T}}) where T = all(is_strongly_convex.(T.parameters))

(g::SeparableSum)(xs::Tuple) = sum(f(x) for (f, x) in zip(g.fs, xs))

prox!(ys::Tuple, fs::Tuple, xs::Tuple, gamma::Number) = sum(prox!(y, f, x, gamma) for (y, f, x) in zip(ys, fs, xs))

prox!(ys::Tuple, fs::Tuple, xs::Tuple, gammas::Tuple) = sum(prox!(y, f, x, gamma) for (y, f, x, gamma) in zip(ys, fs, xs, gammas))

prox!(ys::Tuple, g::SeparableSum, xs::Tuple, gamma) = prox!(ys, g.fs, xs, gamma)

gradient!(grads::Tuple, fs::Tuple, xs::Tuple) = sum(gradient!(grad, f, x) for (grad, f, x) in zip(grads, fs, xs))

gradient!(grads::Tuple, g::SeparableSum, xs::Tuple) = gradient!(grads, g.fs, xs)

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
