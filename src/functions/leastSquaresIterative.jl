### CONCRETE TYPE: ITERATIVE PROX EVALUATION
# prox! is computed using CG on a system with matrix lambda*A'A + I/gamma
# or lambda*AA' + I/gamma, according to which matrix is smaller.

using LinearAlgebra

struct LeastSquaresIterative{N, R <: Real, RC <: RealOrComplex{R}, M, V <: AbstractArray{RC, N}, O} <: LeastSquares
    A::M # m-by-n operator
    b::V # m (by-p)
    lambda::R
    lambdaAtb::V
    shape::Symbol
    S::O
    res::Array{RC, N} # m (by-p)
    res2::Array{RC, N} # m (by-p)
    q::Array{RC, N} # n (by-p)
end

is_prox_accurate(f::LeastSquaresIterative) = false
is_convex(f::LeastSquaresIterative) = f.lambda >= 0
is_concave(f::LeastSquaresIterative) = f.lambda <= 0

function LeastSquaresIterative(A::M, b::V, lambda::R) where {N, R <: Real, RC <: RealOrComplex{R}, M, V <: AbstractArray{RC, N}}
    if size(A, 1) != size(b, 1)
        error("A and b have incompatible dimensions")
    end
    m, n = size(A)
    x_shape = infer_shape_of_x(A, b)
    if m >= n
        shape = :Tall
        S = AcA(A)
        LeastSquaresIterative{N, R, RC, M, V, AcA}(A, b, lambda, lambda*(A'*b), shape, S, zero(b), [], zeros(RC, x_shape))
    else
        shape = :Fat
        S = AAc(A)
        LeastSquaresIterative{N, R, RC, M, V, AAc}(A, b, lambda, lambda*(A'*b), shape, S, zero(b), zero(b), zeros(RC, x_shape))
    end
end

function (f::LeastSquaresIterative{N, R, RC, M, V})(x::AbstractArray{RC, N}) where {N, R, RC, M, V}
    mul!(f.res, f.A, x)
    f.res .-= f.b
    return (f.lambda/2)*norm(f.res, 2)^2
end

function prox!(y::AbstractArray{D, N}, f::LeastSquaresIterative{N, R, RC, M, V}, x::AbstractArray{D, N}, gamma::R=R(1)) where {N, R, RC, M, V, D <: RealOrComplex{R}}
    f.q .= f.lambdaAtb .+ x./gamma
    # two cases: (1) tall A, (2) fat A
    if f.shape == :Tall
        y .= x
        op = ScaleShift(RC(f.lambda), f.S, RC(1)/gamma)
        IterativeSolvers.cg!(y, op, f.q)
    else # f.shape == :Fat
        # y .= gamma*(f.q - lambda*(f.A'*(f.fact\(f.A*f.q))))
        mul!(f.res, f.A, f.q)
        op = ScaleShift(RC(f.lambda), f.S, RC(1)/gamma)
        IterativeSolvers.cg!(f.res2, op, f.res)
        mul!(y, adjoint(f.A), f.res2)
        y .*= -f.lambda
        y .+= f.q
        y .*= gamma
    end
    mul!(f.res, f.A, y)
    f.res .-= f.b
    return (f.lambda/2)*norm(f.res, 2)^2
end

function gradient!(y::AbstractArray{D, N}, f::LeastSquaresIterative{N, R, RC, M, V}, x::AbstractArray{D, N}) where {N, R, RC, M, V, D <: Union{R, Complex{R}}}
    mul!(f.res, f.A, x)
    f.res .-= f.b
    mul!(y, adjoint(f.A), f.res)
    y .*= f.lambda
    return (f.lambda / 2) * real(dot(f.res, f.res))
end

function prox_naive(f::LeastSquaresIterative{N}, x::AbstractArray{D, N}, gamma::R=R(1)) where {N, R, D <: RealOrComplex{R}}
    y = IterativeSolvers.cg(f.lambda*f.A'*f.A + I/gamma, f.lambda*f.A'*f.b + x/gamma)
    fy = (f.lambda/2)*norm(f.A*y-f.b)^2
    return y, fy
end
