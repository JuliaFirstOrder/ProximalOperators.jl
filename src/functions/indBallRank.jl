# indicator of the ball of matrices with (at most) a given rank

using LinearAlgebra
using TSVD

export IndBallRank

"""
**Indicator of rank ball**

	IndBallRank(r=1)

Returns the indicator function of the set of matrices of rank at most `r`:
```math
S = \\{ X : \\mathrm{rank}(X) \\leq r \\},
```
Parameter `r` must be a positive integer.
"""
struct IndBallRank{I <: Integer} <: ProximableFunction
	r::I
	function IndBallRank{I}(r::I) where {I <: Integer}
		if r <= 0
			error("parameter r must be a positive integer")
		else
			new(r)
		end
	end
end

is_set(f::IndBallRank) = true
is_prox_accurate(f::IndBallRank) = false

IndBallRank(r::I=1) where {I <: Integer} = IndBallRank{I}(r)

function (f::IndBallRank)(x::AbstractArray{T, 2}) where {R <: Real, T <: RealOrComplex{R}}
	maxr = minimum(size(x))
	if maxr <= f.r return R(0) end
	U, S, V = tsvd(x, f.r+1)
	# the tolerance in the following line should be customizable
	if S[end]/S[1] <= 1e-7
		return R(0)
	end
	return R(Inf)
end

function prox!(y::AbstractMatrix{T}, f::IndBallRank, x::AbstractMatrix{T}, gamma::R=R(1)) where {R <: Real, T <: RealOrComplex{R}}
	maxr = minimum(size(x))
	if maxr <= f.r
		y .= x
		return R(0)
	end
	U, S, V = tsvd(x, f.r)
	# TODO: the order of the following matrix products should depend on the shape of x
	M = S .* V'
	mul!(y, U, M)
	return R(0)
end

fun_name(f::IndBallRank) = "indicator of the set of rank-r matrices"
fun_dom(f::IndBallRank) = "AbstractArray{Real,2}, AbstractArray{Complex,2}"
fun_expr(f::IndBallRank) = "x ↦ 0 if rank(x) ⩽ r, +∞ otherwise"
fun_params(f::IndBallRank) = "r = $(f.r)"

function prox_naive(f::IndBallRank, x::AbstractMatrix{T}, gamma::R=R(1)) where {R <: Real, T <: RealOrComplex{R}}
	maxr = minimum(size(x))
	if maxr <= f.r
		y = x
		return y, R(0)
	end
	F = svd(x)
	y = F.U[:,1:f.r]*(Diagonal(F.S[1:f.r])*F.V[:,1:f.r]')
	return y, R(0)
end
