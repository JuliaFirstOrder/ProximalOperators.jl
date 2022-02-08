# Precompose with a linear mapping + translation (= affine)

export Precompose

"""
    Precompose(f, L, μ, b)

Return the function
```math
g(x) = f(Lx + b)
```
where ``f`` is a convex function and ``L`` is a linear mapping: this must
satisfy ``LL^* = μI`` for ``μ > 0``. Furthermore, either ``f`` is separable or
parameter `μ` is a scalar, for the `prox` of ``g`` to be computable.

Parameter `L` defines ``L`` through the `mul!` method. Therefore `L` can be an
`AbstractMatrix` for example, but not necessarily.

In this case, `prox` and `prox!` are computed according to Prop. 24.14 in
Bauschke, Combettes "Convex Analysis and Monotone Operator Theory in Hilbert
Spaces", 2nd edition, 2016. The same result is Prop. 23.32 in the 1st edition
of the same book.
"""
struct Precompose{T, R <: Real, C <: Union{R, Complex{R}}, U <: Union{C, AbstractArray{C}}, V <: Union{C, AbstractArray{C}}, M}
    f::T
    L::M
    mu::U
    b::V
    function Precompose{T, R, C, U, V, M}(f::T, L::M, mu::U, b::V) where {T, R <: Real, C <: Union{R, Complex{R}}, U <: Union{R, AbstractArray{R}}, V <: Union{C, AbstractArray{C}}, M}
        if !is_convex(f)
            error("f must be convex")
        end
        if any(mu .<= 0.0)
            error("elements of μ must be positive")
        end
        new(f, L, mu, b)
    end
end

is_prox_accurate(::Type{<:Precompose{T}}) where T = is_prox_accurate(T)
is_convex(::Type{<:Precompose{T}}) where T = is_convex(T)
is_set(::Type{<:Precompose{T}}) where T = is_set(T)
is_singleton(::Type{<:Precompose{T}}) where T = is_singleton(T)
is_cone(::Type{<:Precompose{T}}) where T = is_cone(T)
is_affine(::Type{<:Precompose{T}}) where T = is_affine(T)
is_smooth(::Type{<:Precompose{T}}) where T = is_smooth(T)
is_generalized_quadratic(::Type{<:Precompose{T}}) where T = is_generalized_quadratic(T)
is_strongly_convex(::Type{<:Precompose{T}}) where T = is_strongly_convex(T)

Precompose(f::T, L::M, mu::U, b::V) where {T, R <: Real, C <: Union{R, Complex{R}}, U <: Union{R, AbstractArray{R}}, V <: Union{C, AbstractArray{C}}, M} = Precompose{T, R, C, U, V, M}(f, L, mu, b)

Precompose(f::T, L::M, mu::U) where {T, R <: Real, U <: Union{R, AbstractArray{R}}, M} = Precompose{T, R, R, U, R, M}(f, L, mu, R(0))

function (g::Precompose)(x::T) where {T <: Union{Tuple, AbstractArray}}
    return g.f(g.L*x .+ g.b)
end

function gradient!(y::AbstractArray{T}, g::Precompose, x::AbstractArray{T}) where T <: RealOrComplex
    res = g.L*x .+ g.b
    gradres = similar(res)
    v = gradient!(gradres, g.f, res)
    mul!(y, adjoint(g.L), gradres)
    return v
end

function prox!(y::AbstractArray{C}, g::Precompose, x::AbstractArray{C}, gamma) where {R <: Real, C <: Union{R, Complex{R}}}
    # See Prop. 24.14 in Bauschke, Combettes "Convex Analysis and Monotone Operator Theory in Hilbert Spaces", 2nd ed., 2016.
    # The same result is Prop. 23.32 in the 1st ed. of the same book.
    #
    # This case has an additional translation: if f(x) = h(x + b) then
    #     prox_f(x) = prox_h(x + b) - b
    # Then one can apply the above mentioned result to g(x) = f(Lx).
    #
    res = g.L*x .+ g.b
    proxres = similar(res)
    v = prox!(proxres, g.f, res, g.mu.*gamma)
    proxres .-= res
    proxres ./= g.mu
    mul!(y, adjoint(g.L), proxres)
    y .+= x
    return v
end

function prox_naive(g::Precompose, x::AbstractArray{C}, gamma) where {R <: Real, C <: Union{R, Complex{R}}}
    res = g.L*x .+ g.b
    proxres, v = prox_naive(g.f, res, g.mu .* gamma)
    y = x + g.L'*((proxres .- res)./g.mu)
    return y, v
end

fun_name(f::Precompose) = "Precomposition"
fun_dom(f::Precompose) = fun_dom(f.f)
fun_expr(f::Precompose) = "x ↦ f(L(x) + b)"
