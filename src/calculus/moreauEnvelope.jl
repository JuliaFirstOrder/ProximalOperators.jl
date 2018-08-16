export MoreauEnvelope

"""
**Moreau envelope**

    MoreauEnvelope(f, γ=1.0)

Returns the Moreau envelope (also known as Moreau-Yosida regularization) of function `f` with parameter `γ` (positive), that is
```math
f^γ(x) = \\min_z \\left\\{ f(z) + \\tfrac{1}{2γ}\\|z-x\\|^2 \\right\\}.
```
If ``f`` is convex, then ``f^γ`` is a smooth, convex, lower approximation to ``f``, having the same minima as the original function.
"""

mutable struct MoreauEnvelope{R <: Real, T <: ProximableFunction} <: ProximableFunction
	g::T
	lambda::R
    function MoreauEnvelope{R, T}(g::T, lambda::R) where {R, T}
    	if lambda <= 0 error("parameter lambda must be positive") end
    	new(g, lambda)
    end
end

MoreauEnvelope(g::T, lambda::R=1.0) where {R <: Real, T <: ProximableFunction} = MoreauEnvelope{R, T}(g, lambda)

is_convex(f::MoreauEnvelope) = is_convex(f.g)
is_smooth(f::MoreauEnvelope) = is_convex(f.g)
is_quadratic(f::MoreauEnvelope) = is_generalized_quadratic(f.g)
is_strongly_convex(f::MoreauEnvelope) = is_strongly_convex(f.g)

function (f::MoreauEnvelope)(x::AbstractArray)
	buf = similar(x)
	g_prox = prox!(buf, f.g, x, f.lambda)
	return g_prox + 1/(2*f.lambda)*norm(buf .- x)^2
end

function gradient!(grad::AbstractArray, f::MoreauEnvelope, x::AbstractArray)
	g_prox = prox!(grad, f.g, x, f.lambda)
	grad .= (x .- grad)/f.lambda
	fx = g_prox + (f.lambda/2)*norm(grad)^2
	return fx
end

fun_name(f::MoreauEnvelope,i::Int64) =
"f$(i)(prox{λ$(i),f$(i)}(A$(i)x))+ 1/2 ‖x - prox{λ$(i),f$(i)}(A$(i)x)‖²"

fun_par( f::MoreauEnvelope,i::Int64)  = "λ$i = $(round(f.lambda,3))"
