# indicator of second-order cones

export IndSOC, IndRotatedSOC

"""
    IndSOC()

Return the indicator of the second-order cone (also known as ice-cream cone or Lorentz cone), that is
```math
C = \\left\\{ (t, x) : \\|x\\| \\leq t \\right\\}.
```
"""
struct IndSOC end

function (::IndSOC)(x)
    T = eltype(x)
    # the tolerance in the following line should be customizable
    if isapprox_le(norm(x[2:end]), x[1], atol=eps(T), rtol=sqrt(eps(T)))
        return T(0)
    end
    return T(Inf)
end

is_convex(f::Type{<:IndSOC}) = true
is_cone_indicator(f::Type{<:IndSOC}) = true

function prox!(y, ::IndSOC, x, gamma)
    T = eltype(x)
    @views nx = norm(x[2:end])
    t = x[1]
    if t <= -nx
        y .= T(0)
    elseif t >= nx
        y .= x
    else
        r = T(0.5) * (T(1) + t / nx)
        y[1] = r * nx
        @views y[2:end] .= r .* x[2:end]
    end
    return T(0)
end

function prox_naive(::IndSOC, x, gamma)
    T = eltype(x)
    nx = norm(x[2:end])
    t = x[1]
    if t <= -nx
        y = zero(x)
    elseif t >= nx
        y = x
    else
        y = zero(x)
        r = T(0.5) * (T(1) + t / nx)
        y[1] = r * nx
        y[2:end] .= r .* x[2:end]
    end
    return y, T(0)
end

# ########################
# ROTATED SOC
# ########################

"""
**Indicator of the rotated second-order cone**

        IndRotatedSOC()

Return the indicator of the *rotated* second-order cone (also known as ice-cream cone or Lorentz cone), that is
```math
C = \\left\\{ (p, q, x) : \\|x\\|^2 \\leq 2\\cdot pq, p \\geq 0, q \\geq 0 \\right\\}.
```
"""
struct IndRotatedSOC end

function (::IndRotatedSOC)(x)
    T = eltype(x)
    if isapprox_le(0, x[1], atol=eps(T), rtol=sqrt(eps(T))) &&
        isapprox_le(0, x[2], atol=eps(T), rtol=sqrt(eps(T))) &&
        isapprox_le(norm(x[3:end])^2, 2*x[1]*x[2], atol=eps(T), rtol=sqrt(eps(T)))
        return T(0)
    end
    return T(Inf)
end

is_convex(f::IndRotatedSOC) = true
is_set_indicator(f::IndRotatedSOC) = true

function prox!(y, ::IndRotatedSOC, x, gamma)
    T = eltype(x)
    # sin(pi/4) = cos(pi/4) = 0.7071067811865475
    # rotate x ccw by pi/4
    x1 = 0.7071067811865475*x[1] + 0.7071067811865475*x[2]
    x2 = 0.7071067811865475*x[1] - 0.7071067811865475*x[2]
    # project rotated x onto SOC
    @views nx = sqrt(x2^2+norm(x[3:end])^2)
    t = x1
    if t <= -nx
        y .= T(0)
    elseif t >= nx
        y[1] = x1
        y[2] = x2
        @views y[3:end] .= x[3:end]
    else
        r = T(0.5) * (T(1) + t / nx)
        y[1] = r * nx
        y[2] = r * x2
        @views y[3:end] = r .* x[3:end]
    end
    # rotate back y cw by pi/4
    y1 = 0.7071067811865475*y[1] + 0.7071067811865475*y[2]
    y2 = 0.7071067811865475*y[1] - 0.7071067811865475*y[2]
    y[1] = y1
    y[2] = y2
    return T(0)
end

function prox_naive(::IndRotatedSOC, x, gamma)
    g = IndSOC()
    z = copy(x)
    z[1] = 0.7071067811865475*x[1] + 0.7071067811865475*x[2]
    z[2] = 0.7071067811865475*x[1] - 0.7071067811865475*x[2]
    y, = prox_naive(g, z, gamma)
    y1 = 0.7071067811865475*y[1] + 0.7071067811865475*y[2]
    y2 = 0.7071067811865475*y[1] - 0.7071067811865475*y[2]
    y[1] = y1
    y[2] = y2
    return y, eltype(x)(0)
end
