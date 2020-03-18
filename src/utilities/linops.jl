# Utility operators for computing prox iteratively, e.g. using CG
#
# The idea is that whatever type has the A_mul_B!, Ac_mul_B!, size and eltype
# methods implemented is a linear operator.

import Base: *, size, eltype
import LinearAlgebra: mul!

abstract type LinOp end

function (*)(Op::LinOp, x)
  # Is this the right thing to do?
  # Or maybe just: y = zeros(eltype(x), size(Op, 1))
  y = zeros(promote_type(eltype(Op), eltype(x)), size(Op, 1))
  mul!(y, Op, x)
end

size(Op::LinOp, i::Integer) = i <= 2 ? size(Op)[i] : 1

# AAc (Gram matrix)

struct AAc{M, T} <: LinOp
  A::M
  buf::AbstractArray{T}
  function AAc{M, T}(A::M) where {M, T}
    new(A, zeros(T, size(A, 2)))
  end
end

AAc(A::M) where {M} = AAc{M, eltype(A)}(A)

function mul!(y, Op::AAc, x)
  mul!(Op.buf, adjoint(Op.A), x)
  mul!(y, Op.A, Op.buf)
end

mul!(y, Op::Adjoint{AAc}, x) = mul!(y, adjoint(Op), x)

size(Op::AAc) = size(Op.A, 1), size(Op.A, 1)
eltype(Op::AAc) = eltype(Op.A)

# AcA (Covariance matrix)

struct AcA{M, T} <: LinOp
  A::M
  buf::AbstractArray{T}
  function AcA{M, T}(A::M) where {M, T}
    new(A, zeros(T, size(A, 1)))
  end
end

AcA(A::M) where {M} = AcA{M, eltype(A)}(A)

function mul!(y, Op::AcA, x)
  mul!(Op.buf, Op.A, x)
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
      error("type of alpha, rho is incompatible with A")
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
