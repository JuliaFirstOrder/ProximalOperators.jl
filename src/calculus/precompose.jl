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
struct Precompose{T, M, U, V}
    f::T
    L::M
    mu::U
    b::V
    function Precompose{T, M, U, V}(f::T, L::M, mu::U, b::V) where {T, M, U, V}
        if !is_convex(f)
            error("f must be convex")
        end
        if any(mu .<= 0)
            error("elements of μ must be positive")
        end
        new(f, L, mu, b)
    end
end

is_proximable(::Type{<:Precompose{T}}) where T = is_proximable(T)
is_convex(::Type{<:Precompose{T}}) where T = is_convex(T)
is_set_indicator(::Type{<:Precompose{T}}) where T = is_set_indicator(T)
is_singleton_indicator(::Type{<:Precompose{T}}) where T = is_singleton_indicator(T)
is_cone_indicator(::Type{<:Precompose{T}}) where T = is_cone_indicator(T)
is_affine_indicator(::Type{<:Precompose{T}}) where T = is_affine_indicator(T)
is_smooth(::Type{<:Precompose{T}}) where T = is_smooth(T)
is_locally_smooth(::Type{<:Precompose{T}}) where T = is_locally_smooth(T)
is_generalized_quadratic(::Type{<:Precompose{T}}) where T = is_generalized_quadratic(T)
is_strongly_convex(::Type{<:Precompose{T}}) where T = is_strongly_convex(T)

Precompose(f::T, L::M, mu::U, b::V) where {T, M, U, V} = Precompose{T, M, U, V}(f, L, mu, b)

Precompose(f::T, L::M, mu::U) where {T, M, U} = Precompose(f, L, mu, 0)

function (g::Precompose)(x)
    return g.f(g.L * x .+ g.b)
end

function gradient!(y, g::Precompose, x)
    res = g.L*x .+ g.b
    gradres = similar(res)
    v = gradient!(gradres, g.f, res)
    mul!(y, adjoint(g.L), gradres)
    return v
end

function prox!(y, g::Precompose, x, gamma)
    # See Prop. 24.14 in Bauschke, Combettes
    # "Convex Analysis and Monotone Operator Theory in Hilbert Spaces",
    # 2nd ed., 2016.
    # 
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

function prox_naive(g::Precompose, x, gamma)
    res = g.L*x .+ g.b
    proxres, v = prox_naive(g.f, res, g.mu .* gamma)
    y = x + g.L'*((proxres .- res)./g.mu)
    return y, v
end
