# Utility operators for computing prox iteratively, e.g. using CG

import Base: *, size, eltype
import LinearAlgebra: mul!

abstract type LinOp end

infer_shape_of_y(Op, ::AbstractVector) = (size(Op, 1), )
infer_shape_of_y(Op, x::AbstractMatrix) = (size(Op, 1), size(x, 2))

function (*)(Op::LinOp, x)
    y = zeros(promote_type(eltype(Op), eltype(x)), infer_shape_of_y(Op, x))
    mul!(y, Op, x)
end

size(Op::LinOp, i::Integer) = i <= 2 ? size(Op)[i] : 1

# AAc (Gram matrix)

struct AAc{M, T} <: LinOp
    A::M
    buf::T
end

function AAc(A::M, input_shape::Tuple) where M
    buffer_shape = (size(A, 2), input_shape[2:end]...)
    buffer = zeros(eltype(A), buffer_shape)
    AAc(A, buffer)
end

function mul!(y, Op::AAc, x)
    if Op.buf === nothing
        Op.buf = adjoint(Op.A) * x
    else
        mul!(Op.buf, adjoint(Op.A), x)
    end
    mul!(y, Op.A, Op.buf)
end

mul!(y, Op::Adjoint{AAc}, x) = mul!(y, adjoint(Op), x)

size(Op::AAc) = size(Op.A, 1), size(Op.A, 1)
eltype(Op::AAc) = eltype(Op.A)

# AcA (Covariance matrix)

struct AcA{M, T} <: LinOp
    A::M
    buf::T
end

function AcA(A::M, input_shape::Tuple) where M
    buffer_shape = (size(A, 1), input_shape[2:end]...)
    buffer = zeros(eltype(A), buffer_shape)
    AcA(A, buffer)
end

function mul!(y, Op::AcA, x)
    if Op.buf === nothing
        Op.buf = Op.A * x
    else
        mul!(Op.buf, Op.A, x)
    end
    mul!(y, adjoint(Op.A), Op.buf)
end

mul!(y, Op::Adjoint{AcA}, x) = mul!(y, adjoint(Op), x)

size(Op::AcA) = size(Op.A, 2), size(Op.A, 2)
eltype(Op::AcA) = eltype(Op.A)

# Shifted symmetric linear operator

struct ScaleShift{M, T} <: LinOp
    alpha::T
    A::M
    rho::T
    function ScaleShift{M, T}(alpha::T, A::M, rho::T) where {M, T}
        if eltype(A) != T
            error("type of alpha, rho ($T) is different from that of A ($(eltype(A)))")
        end
        new(alpha, A, rho)
    end
end

ScaleShift(alpha::T, A::M, rho::T) where {M, T} = ScaleShift{M, T}(alpha, A, rho)

function mul!(y, Op::ScaleShift, x)
    mul!(y, Op.A, x)
    y .*= Op.alpha
    y .+= Op.rho .* x
end

function mul!(y, Op::Adjoint{ScaleShift}, x)
    mul!(y, adjoint(Op.A), x)
    y .*= Op.alpha
    y .+= Op.rho .* x
end

size(Op::ScaleShift) = size(Op.A, 2), size(Op.A, 2)
eltype(Op::ScaleShift) = eltype(Op.A)
