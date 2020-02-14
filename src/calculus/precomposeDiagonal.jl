# Precompose with diagonal scaling and translation

export PrecomposeDiagonal

"""
**Precomposition with diagonal scaling/translation**

    PrecomposeDiagonal(f, a, b)

Returns the function
```math
g(x) = f(\\mathrm{diag}(a)x + b)
```
where ``f`` is a convex function. Function ``f`` must be separable,
or `a` must be a scalar, for the `prox` of ``g`` to be computable.
Parametes `a` and `b` can be arrays of multiple dimensions, according to
the shape/size of the input `x` that will be provided to the function:
the way the above expression for ``g`` should be thought of, is
`g(x) = f(a.*x + b)`.
"""
struct PrecomposeDiagonal{T <: ProximableFunction, R <: Union{Real, AbstractArray}, S <: Union{Real, AbstractArray}} <: ProximableFunction
    f::T
    a::R
    b::S
    function PrecomposeDiagonal{T,R,S}(f::T, a::R, b::S) where {T <: ProximableFunction, R <: Union{Real, AbstractArray}, S <: Union{Real, AbstractArray}}
        if !is_convex(f)
            error("`f` must be convex")
        end
        if !(eltype(a) <: Real)
            error("`a` must have real elements")
        end
        if any(a == 0.0)
            error("elements of `a` must be nonzero")
        else
            new(f, a, b)
        end
    end
end

is_separable(f::PrecomposeDiagonal) = is_separable(f.f)
is_prox_accurate(f::PrecomposeDiagonal) = is_prox_accurate(f.f)
is_convex(f::PrecomposeDiagonal) = is_convex(f.f)
is_set(f::PrecomposeDiagonal) = is_set(f.f)
is_singleton(f::PrecomposeDiagonal) = is_singleton(f.f)
is_cone(f::PrecomposeDiagonal) = is_cone(f.f)
is_affine(f::PrecomposeDiagonal) = is_affine(f.f)
is_smooth(f::PrecomposeDiagonal) = is_smooth(f.f)
is_quadratic(f::PrecomposeDiagonal) = is_quadratic(f.f)
is_generalized_quadratic(f::PrecomposeDiagonal) = is_generalized_quadratic(f.f)
is_strongly_convex(f::PrecomposeDiagonal) = is_strongly_convex(f.f)

PrecomposeDiagonal(f::T, a::S=1.0, b::S=0.0) where {T <: ProximableFunction, S <: Real} = PrecomposeDiagonal{T, S, S}(f, a, b)

PrecomposeDiagonal(f::T, a::R, b::S=0.0) where {T <: ProximableFunction, R <: AbstractArray, S <: Real} = PrecomposeDiagonal{T, R, S}(f, a, b)

PrecomposeDiagonal(f::T, a::R, b::S) where {T <: ProximableFunction, R <: Union{AbstractArray, Real}, S <: AbstractArray} = PrecomposeDiagonal{T, R, S}(f, a, b)

function (g::PrecomposeDiagonal)(x::AbstractArray{T}) where T <: RealOrComplex
    return g.f((g.a).*x .+ g.b)
end

function gradient!(y::AbstractArray{T}, g::PrecomposeDiagonal, x::AbstractArray{T}) where T <: RealOrComplex
    z = g.a .* x .+ g.b
    v = gradient!(y, g.f, z)
    y .*= g.a
    return v
end

function prox!(y::AbstractArray{T}, g::PrecomposeDiagonal, x::AbstractArray{T}, gamma::Union{R, AbstractArray{R}}=R(1)) where {R <: Real, T <: RealOrComplex{R}}
    z = g.a .* x .+ g.b
    v = prox!(y, g.f, z, (g.a .* g.a) .* gamma)
    y .-= g.b
    y ./= g.a
    return v
end

function prox_naive(g::PrecomposeDiagonal, x::AbstractArray{T}, gamma::Union{R, AbstractArray{R}}=R(1)) where {R <: Real, T <: RealOrComplex{R}}
    z = g.a .* x .+ g.b
    y, fy = prox_naive(g.f, z, (g.a .* g.a) .* gamma)
    return (y .- g.b)./g.a, fy
end

fun_name(f::PrecomposeDiagonal) = string("Precomposition by affine diagonal mapping of ", fun_name(f.f))
fun_dom(f::PrecomposeDiagonal) = fun_dom(f.f)
fun_expr(f::PrecomposeDiagonal) = "x ↦ f(diag(a)*x + b)"
fun_params(f::PrecomposeDiagonal) = string("f(x) = ", fun_expr(f.f), ", a = ", length(f.a) == 1 ? string(f.a[1]) : string(typeof(f.a)), ", b = ", length(f.b) == 1 ? string(f.b[1]) : string(typeof(f.b)))
