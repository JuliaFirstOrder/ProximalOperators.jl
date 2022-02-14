### CONCRETE TYPE: ITERATIVE PROX EVALUATION

using LinearAlgebra
using IterativeSolvers

struct QuadraticIterative{M, V} <: Quadratic
    Q::M
    q::V
    temp::V
end

is_prox_accurate(f::Type{<:QuadraticIterative}) = false

function QuadraticIterative(Q::M, q::V) where {M, V}
    if size(Q, 1) != size(Q, 2) || length(q) != size(Q, 2)
        error("Q must be squared and q must be compatible with Q")
    end
    QuadraticIterative{M, V}(Q, q, similar(q))
end

function (f::QuadraticIterative)(x)
    mul!(f.temp, f.Q, x)
    return 0.5*dot(x, f.temp) + dot(x, f.q)
end

function prox!(y, f::QuadraticIterative{M, V}, x, gamma) where {M, V}
    R = eltype(M)
    y .= x
    f.temp .= x./gamma .- f.q
    op = ScaleShift(R(1), f.Q, R(1)/gamma)
    IterativeSolvers.cg!(y, op, f.temp)
    mul!(f.temp, f.Q, y)
    fy = 0.5*dot(y, f.temp) + dot(y, f.q)
    return fy
end

function gradient!(y, f::QuadraticIterative, x)
    mul!(y, f.Q, x)
    y .+= f.q
    return 0.5*(dot(x, y) + dot(x, f.q))
end

function prox_naive(f::QuadraticIterative, x, gamma)
    y = IterativeSolvers.cg(gamma*f.Q + I, x - gamma*f.q)
    fy = 0.5*dot(y, f.Q*y) + dot(y, f.q)
    return y, fy
end
