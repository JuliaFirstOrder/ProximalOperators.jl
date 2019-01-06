# indicator of a point

export IndPoint

"""
**Indicator of a singleton**

    IndPoint(p=0.0)

Returns the indicator of the set
```math
C = \\{p \\}.
```
Parameter `p` can be a scalar, in which case the unique element of `S` has uniform coefficients.
"""
struct IndPoint{R <: Real, C <: Union{R, Complex{R}}, T <: Union{C, AbstractArray{C}}} <: ProximableFunction
    p::T
    function IndPoint{R, C, T}(p::T) where {R, C, T}
        new(p)
    end
end

is_separable(f::IndPoint) = true
is_convex(f::IndPoint) = true
is_singleton(f::IndPoint) = true
is_cone(f::IndPoint) = norm(f.p) == 0
is_affine(f::IndPoint) = true

IndPoint(p::T=0.0) where {R <: Real, C <: Union{R, Complex{R}}, T <: Union{C, AbstractArray{C}}} = IndPoint{R, C, T}(p)

function (f::IndPoint{R, C})(x::AbstractArray{C}) where {R, C}
    if all(x .≈ f.p)
        return R(0)
    end
    return R(Inf)
end

function prox!(y::AbstractArray{C}, f::IndPoint{R, C}, x::AbstractArray{C}, gamma=R(1)) where {R, C}
    y .= f.p
    return R(0)
end

fun_name(f::IndPoint) = "indicator of a point"
fun_dom(f::IndPoint) = "AbstractArray{Real}, AbstractArray{Complex}"
fun_expr(f::IndPoint) = "x ↦ 0 if x = p, +∞ otherwise"
fun_params(f::IndPoint) =
    string( "p = ", typeof(f.p) <: AbstractArray ? string(typeof(f.p), " of size ", size(f.p)) : f.p, ", ")

function prox_naive(f::IndPoint{R, C}, x::AbstractArray{C}, gamma=R(1)) where {R, C}
    y = similar(x)
    y .= f.p
    return y, R(0)
end
