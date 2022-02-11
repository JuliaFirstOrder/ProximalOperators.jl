# Precompose with diagonal scaling and translation

export PrecomposeDiagonal

"""
    PrecomposeDiagonal(f, a, b)

Return the function
```math
g(x) = f(\\mathrm{diag}(a)x + b)
```
Function ``f`` must be convex and separable, or `a` must be a scalar, for the
`prox` of ``g`` to be computable. Parametes `a` and `b` can be arrays of
multiple dimensions, according to the shape/size of the input `x` that will be
provided to the function: the way the above expression for ``g`` should be
thought of, is `g(x) = f(a.*x + b)`.
"""
struct PrecomposeDiagonal{T, R, S}
    f::T
    a::R
    b::S
    function PrecomposeDiagonal{T,R,S}(f::T, a::R, b::S) where {T, R, S}
        if R <: AbstractArray && !(is_convex(f) && is_separable(f))
            error("`f` must be convex and separable since `a` is of type $(R)")
        end
        if any(a == 0)
            error("elements of `a` must be nonzero")
        else
            new(f, a, b)
        end
    end
end

is_separable(::Type{<:PrecomposeDiagonal{T}}) where T = is_separable(T)
is_prox_accurate(::Type{<:PrecomposeDiagonal{T}}) where T = is_prox_accurate(T)
is_convex(::Type{<:PrecomposeDiagonal{T}}) where T = is_convex(T)
is_set(::Type{<:PrecomposeDiagonal{T}}) where T = is_set(T)
is_singleton(::Type{<:PrecomposeDiagonal{T}}) where T = is_singleton(T)
is_cone(::Type{<:PrecomposeDiagonal{T}}) where T = is_cone(T)
is_affine(::Type{<:PrecomposeDiagonal{T}}) where T = is_affine(T)
is_smooth(::Type{<:PrecomposeDiagonal{T}}) where T = is_smooth(T)
is_generalized_quadratic(::Type{<:PrecomposeDiagonal{T}}) where T = is_generalized_quadratic(T)
is_strongly_convex(::Type{<:PrecomposeDiagonal{T}}) where T = is_strongly_convex(T)

PrecomposeDiagonal(f::T, a::S=1, b::S=0) where {T, S <: Real} = PrecomposeDiagonal{T, S, S}(f, a, b)

PrecomposeDiagonal(f::T, a::R, b::S=0) where {T, R <: AbstractArray, S <: Real} = PrecomposeDiagonal{T, R, S}(f, a, b)

PrecomposeDiagonal(f::T, a::R, b::S) where {T, R <: Union{AbstractArray, Real}, S <: AbstractArray} = PrecomposeDiagonal{T, R, S}(f, a, b)

function (g::PrecomposeDiagonal)(x)
    return g.f(g.a .* x .+ g.b)
end

function gradient!(y, g::PrecomposeDiagonal, x)
    z = g.a .* x .+ g.b
    v = gradient!(y, g.f, z)
    y .*= g.a
    return v
end

function prox!(y, g::PrecomposeDiagonal, x, gamma)
    z = g.a .* x .+ g.b
    v = prox!(y, g.f, z, (g.a .* g.a) .* gamma)
    y .-= g.b
    y ./= g.a
    return v
end

function prox_naive(g::PrecomposeDiagonal, x, gamma)
    z = g.a .* x .+ g.b
    y, fy = prox_naive(g.f, z, (g.a .* g.a) .* gamma)
    return (y .- g.b)./g.a, fy
end

fun_name(f::PrecomposeDiagonal) = string("Precomposition by affine diagonal mapping of ", fun_name(f.f))
fun_dom(f::PrecomposeDiagonal) = fun_dom(f.f)
fun_expr(f::PrecomposeDiagonal) = "x ↦ f(diag(a)*x + b)"
fun_params(f::PrecomposeDiagonal) = string("f(x) = ", fun_expr(f.f), ", a = ", length(f.a) == 1 ? string(f.a[1]) : string(typeof(f.a)), ", b = ", length(f.b) == 1 ? string(f.b[1]) : string(typeof(f.b)))
