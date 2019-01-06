### CONCRETE TYPE: ITERATIVE PROX EVALUATION

struct IndAffineIterative{R <: Real, T <: RealOrComplex{R}, M <: AbstractMatrix{T}, V <: AbstractVector{T}} <: IndAffine
	A::M
	b::V
	res::V
	maxit::Integer
	tol::R
	function IndAffineIterative{R, T, M, V}(A::M, b::V) where {R <: Real, T <: RealOrComplex{R}, M <: AbstractMatrix{T}, V <: AbstractVector{T}}
		if size(A,1) > size(A,2)
			error("A must be full row rank")
		end
		normrowsinv = 1 ./ vec(sqrt.(sum(abs2.(A); dims=2)))
		A = normrowsinv.*A # normalize rows of A
		b = normrowsinv.*b # and b accordingly
		new(A, b, similar(b), 1000, 1e-8)
	end
end

is_prox_accurate(f::IndAffineIterative) = false
is_cone(f::IndAffineIterative) = norm(f.b) == 0.0

IndAffineIterative(A::M, b::V) where {R <: Real, T <: RealOrComplex{R}, M <: AbstractMatrix{T}, V <: AbstractVector{T}} = IndAffineIterative{R, T, M, V}(A, b)

function (f::IndAffineIterative{R, T, M, V})(x::V) where {R, T, M, V}
	mul!(f.res, f.A, x)
	f.res .= f.b .- f.res
	# the tolerance in the following line should be customizable
	if norm(f.res, Inf) <= f.tol
		return R(0)
	end
	return typemax(R)
end

function prox!(y::V, f::IndAffineIterative{R, T, M, V}, x::V, gamma::R=R(1)) where {R, T, M, V}
	# Von Neumann's alternating projections
	m = size(f.A, 1)
	y .= x
	for k = 1:f.maxit
		maxres = R(0)
		for i = 1:m
			resi = (f.b[i] - dot(f.A[i,:], y))
			y .= y + resi*f.A[i,:] # no need to divide: rows of A are normalized
			absresi = resi > 0 ? resi : -resi
			maxres = absresi > maxres ? absresi : maxres
		end
		if maxres < f.tol
			break
		end
	end
	return R(0)
end

function prox_naive(f::IndAffineIterative, x::AbstractArray{T,1}, gamma::R=R(1)) where {R <: Real, T <: RealOrComplex{R}}
	y = x + f.A'*((f.A*f.A')\(f.b - f.A*x))
	return y, R(0)
end
