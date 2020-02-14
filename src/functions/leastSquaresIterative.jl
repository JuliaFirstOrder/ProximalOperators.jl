### CONCRETE TYPE: ITERATIVE PROX EVALUATION
# prox! is computed using CG on a system with matrix A'A + I/(lambda*gamma)
# or AA' + I/(lambda*gamma), according to which matrix is smaller.

using LinearAlgebra

struct LeastSquaresIterative{R <: Real, RC <: RealOrComplex{R}, M, V <: AbstractVector{RC}, O} <: LeastSquares
    A::M # m-by-n operator
    b::V
    Atb::V
    lambda::R
    shape::Symbol
    S::O
    res::Vector{RC} # m-sized buffer
    res2::Vector{RC} # m-sized buffer
    q::Vector{RC} # n-sized buffer
end

is_prox_accurate(f::LeastSquaresIterative) = false
is_convex(f::LeastSquaresIterative) = f.lambda > 0
is_concave(f::LeastSquaresIterative) = f.lambda < 0

function LeastSquaresIterative(A::M, b::V, lambda::R) where {R <: Real, RC <: RealOrComplex{R}, M, V <: AbstractVector{RC}}
    if size(A, 1) != length(b)
        error("A and b have incompatible dimensions")
    end
    if lambda == 0
        error("lambda must nonzero")
    end
    m, n = size(A)
    if m >= n
        shape = :Tall
        S = AcA(A)
        LeastSquaresIterative{R, RC, M, V, AcA}(A, b, A'*b, lambda, shape, S, zeros(RC, m), [], zeros(RC, n))
    else
        shape = :Fat
        S = AAc(A)
        LeastSquaresIterative{R, RC, M, V, AAc}(A, b, A'*b, lambda, shape, S, zeros(RC, m), zeros(RC, m), zeros(RC, n))
    end
end

function (f::LeastSquaresIterative{R, RC, M, V})(x::AbstractVector{RC}) where {R, RC, M, V}
    mul!(f.res, f.A, x)
    f.res .-= f.b
    return (f.lambda/2)*norm(f.res, 2)^2
end

function prox!(y::AbstractVector{D}, f::LeastSquaresIterative{R, RC, M, V}, x::AbstractVector{D}, gamma::R=R(1)) where {R, RC, M, V, D <: RealOrComplex{R}}
    lamgam = f.lambda*gamma
    f.q .= f.Atb .+ x./lamgam
    # two cases: (1) tall A, (2) fat A
    if f.shape == :Tall
        y .= x
        op = Shift(f.S, R(1)/lamgam)
        IterativeSolvers.cg!(y, op, f.q)
    else # f.shape == :Fat
        # y .= lamgam*(f.q - (f.A'*(f.fact\(f.A*f.q))))
        mul!(f.res, f.A, f.q)
        op = Shift(f.S, R(1)/lamgam)
        IterativeSolvers.cg!(f.res2, op, f.res)
        mul!(y, adjoint(f.A), f.res2)
        y .-= f.q
        y .*= -lamgam
    end
    mul!(f.res, f.A, y)
    f.res .-= f.b
    return (f.lambda/2)*norm(f.res, 2)^2
end

function gradient!(y::AbstractVector{D}, f::LeastSquaresIterative{R, RC, M, V}, x::AbstractVector{D}) where {R, RC, M, V, D <: Union{R, Complex{R}}}
    mul!(f.res, f.A, x)
    f.res .-= f.b
    mul!(y, adjoint(f.A), f.res)
    y .*= f.lambda
    fy = (f.lambda/2)*dot(f.res, f.res)
end

function prox_naive(f::LeastSquaresIterative, x::AbstractVector{D}, gamma::R=R(1)) where {R, D <: RealOrComplex{R}}
    lamgam = f.lambda*gamma
    y = IterativeSolvers.cg(f.A'*f.A + I/lamgam, f.Atb + x/lamgam)
    fy = (f.lambda/2)*norm(f.A*y-f.b)^2
    return y, fy
end
