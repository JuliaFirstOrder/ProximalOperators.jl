# indicator of the L1 norm ball with given radius

"""
  IndBallL1(r::Real=1.0)

Returns the function `g = ind{x : ‖x‖_1 ⩽ r}`, for a real parameter `r > 0`.
"""

immutable IndBallL1 <: IndicatorConvex
  r::Real
  function IndBallL1(r::Real=1.0)
    if r <= 0
      error("parameter r must be positive")
    else
      new(r)
    end
  end
end

@compat function (f::IndBallL1){T <: RealOrComplex}(x::AbstractArray{T,1})
  if vecnorm(x,1) - f.r > 1e-14
    return +Inf
  end
  return 0.0
end

function prox!{T <: RealOrComplex}(f::IndBallL1, x::AbstractArray{T,1}, y::AbstractArray{T,1}, gamma::Real=1.0)
  #TODO: a faster algorithm
  if f.r - vecnorm(x,1) > -1e-14;
    y[:] = x[:]
    return 0.0
  else #do a projection of abs(y) onto simplex then recover signs
    n = length(x)
    p = abs(x)
    sort!(p,rev = true)

    s = 0.0
    @inbounds for i = 1:n-1
      s = s + p[i]
      tmax = (s - f.r)/i
      if tmax >= p[i+1]
        for it in eachindex(x)
          y[it] = sign(x[it]).*max(abs(x[it])-tmax, 0.0)
        end
        return 0.0
      end
    end
    tmax = (s + p[n] - f.r)/n
    @inbounds for i in eachindex(x)
      y[i] = sign(x[i]).*max(abs(x[i])-tmax, 0.0)
    end
    return 0.0
  end

end

fun_name(f::IndBallL1) = "indicator of an L1 norm ball"
fun_type(f::IndBallL1) = "Array{Complex} → Real ∪ {+∞}"
fun_expr(f::IndBallL1) = "x ↦ 0 if ‖x‖_1 ⩽ r, +∞ otherwise"
fun_params(f::IndBallL1) = "r = $(f.r)"

function prox_naive{T <: RealOrComplex}(f::IndBallL1, x::AbstractArray{T,1}, gamma::Real=1.0)
  if vecnorm(x,1) <= f.r
    return x, 0.0;
  else   #do a simple bisection (aka binary search) on λ
    L = 0.0;
    U = maxabs(x);
    λ = L;
    v = 0.0;
    maxIter::Int = 120;

    for iter in range(1,maxIter)
      λ = 0.5*(L + U)
      v = 0.0;
     @inbounds @simd for i in eachindex(x)
          v = v + max(abs(x[i]) - λ,0.);
      end
      #modify lower or upper bound
      (v < f.r) ?  U = λ : L = λ
      # exit condition
      if  abs(L - U) < 1e-15
        break;
      end
    end

    return sign(x).*max(0.0, abs(x)-λ), 0.0
  end
end
