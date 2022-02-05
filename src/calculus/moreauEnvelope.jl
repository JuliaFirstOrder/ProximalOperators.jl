export MoreauEnvelope

"""
**Moreau envelope**

    MoreauEnvelope(f, γ=1)

Returns the Moreau envelope (also known as Moreau-Yosida regularization) of function `f` with parameter `γ` (positive), that is
```math
f^γ(x) = \\min_z \\left\\{ f(z) + \\tfrac{1}{2γ}\\|z-x\\|^2 \\right\\}.
```
If ``f`` is convex, then ``f^γ`` is a smooth, convex, lower approximation to ``f``, having the same minima as the original function.
"""
struct MoreauEnvelope{R, T}
    g::T
    lambda::R
    function MoreauEnvelope{R, T}(g::T, lambda::R) where {R, T}
        if lambda <= 0 error("parameter lambda must be positive") end
        new(g, lambda)
    end
end

MoreauEnvelope(g::T, lambda::R=1) where {R, T} = MoreauEnvelope{R, T}(g, lambda)

is_convex(::Type{MoreauEnvelope{R, T}}) where {R, T} = is_convex(T)
is_smooth(::Type{MoreauEnvelope{R, T}}) where {R, T} = is_convex(T)
is_generalized_quadratic(::Type{MoreauEnvelope{R, T}}) where {R, T} = is_generalized_quadratic(T)
is_strongly_convex(::Type{MoreauEnvelope{R, T}}) where {R, T} = is_strongly_convex(T)

function (f::MoreauEnvelope)(x)
    R = eltype(x)
    buf = similar(x)
    g_prox = prox!(buf, f.g, x, f.lambda)
    return g_prox + R(1) / (2 * f.lambda) * norm(buf .- x)^2
end

function gradient!(grad, f::MoreauEnvelope, x)
    R = eltype(x)
    g_prox = prox!(grad, f.g, x, f.lambda)
    grad .= (x .- grad)./f.lambda
    fx = g_prox + f.lambda / R(2) * norm(grad)^2
    return fx
end

function prox!(u, f::MoreauEnvelope, x, gamma)
    # See: Thm. 6.63 in A. Beck, "First-Order Methods in Optimization", MOS-SIAM Series on Optimization, SIAM, 2017
    R = eltype(x)
    gamma_lambda = gamma + f.lambda
    y, fy = prox(f.g, x, gamma_lambda)
    alpha = gamma / gamma_lambda
    u .= ((1 - alpha) .* x) .+ (alpha .* y)
    return fy + R(1) / (2 * f.lambda) * norm(u .- y)^2
end

# NOTE the following is just so we can use certain test helpers
# TODO properly implement the following
prox_naive(f::MoreauEnvelope, x, gamma) = prox(f, x, gamma)
