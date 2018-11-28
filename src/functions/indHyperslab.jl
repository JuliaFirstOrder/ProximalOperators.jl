# indicator of a hyperslab

export IndHyperslab

"""
**Indicator of a hyperslab**

    IndHyperslab(low, a, upp)

For an array `a` and scalars `low` and `upp`, returns the indicator of set
```math
S = \\{x : low \\leq \\langle a,x \\rangle \\leq upp \\}.
```
"""
struct IndHyperslab{R <: Real, T <: AbstractArray{R}} <: ProximableFunction
  low::R
  a::T
  upp::R
  norm_a::R
  function IndHyperslab{R, T}(low::R, a::T, upp::R) where {R <: Real, T <: AbstractArray{R}}
    norm_a = norm(a)
    if (norm_a == 0 && (upp < 0 || low > 0)) || upp < low
        error("function is improper")
    end
    new(low, a, upp, norm_a)
  end
end

IndHyperslab(low::R, a::T, upp::R) where {R <: Real, T <: AbstractArray{R}} = IndHyperslab{R, T}(low, a, upp)

is_convex(f::IndHyperslab) = true
is_set(f::IndHyperslab) = true
is_cone(f::IndHyperslab{R}) where R =
    (f.low == f.upp == 0) ||
    (f.low == 0 && f.upp == Inf) ||
    (f.low == -Inf && f.upp == 0) ||
    (f.low == -Inf && f.upp == Inf)

function (f::IndHyperslab{R})(x::AbstractArray{R}) where R
  s = dot(f.a, x)
  # if f.low <= s <= f.upp
  # tol = (R(1) + abs(s))*eps(R)
  if f.low - s <= eps(R)*f.norm_a*(1 + abs(f.low)) && s - f.upp <= eps(R)*f.norm_a*(1 + abs(f.upp))
    return zero(R)
  end
  return R(Inf)
end

function prox!(y::AbstractArray{R}, f::IndHyperslab{R}, x::AbstractArray{R}, gamma::R=one(R)) where R
  s = dot(f.a, x)
  if s < f.low && f.norm_a > 0
    y .= x .- ((s - f.low)/f.norm_a^2) .* f.a
  elseif s > f.upp && f.norm_a > 0
    y .= x .- ((s - f.upp)/f.norm_a^2) .* f.a
  else
    copyto!(y, x)
  end
  return zero(R)
end

function prox_naive(f::IndHyperslab{R}, x::AbstractArray{R}, gamma::R=one(R)) where R
  s = dot(f.a, x)
  if s < f.low && f.norm_a > 0
    return x - ((s - f.low)/norm(f.a)^2) * f.a, R(0)
  elseif s > f.upp && f.norm_a > 0
    return x - ((s - f.upp)/norm(f.a)^2) * f.a, R(0)
  else
    return x, R(0)
  end
end
