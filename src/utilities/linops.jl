# Utility operators for computing prox iteratively, e.g. using CG
#
# The idea is that whatever type has the A_mul_B!, Ac_mul_B!, size and eltype
# methods implemented is a linear operator.

import Base: A_mul_B!, Ac_mul_B!, *, size, eltype

abstract type LinOp end

function (*)(Op::LinOp, x)
  # Is this the right thing to do?
  # Or maybe just: y = zeros(eltype(x), size(Op, 1))
  y = zeros(promote_type(eltype(Op), eltype(x)), size(Op, 1))
  A_mul_B!(y, Op, x)
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

function A_mul_B!(y, Op::AAc, x)
  Ac_mul_B!(Op.buf, Op.A, x)
  A_mul_B!(y, Op.A, Op.buf)
end

Ac_mul_B!(y, Op::AAc, x) = A_mul_B!(y, Op::AAc, x)

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

function A_mul_B!(y, Op::AcA, x)
  A_mul_B!(Op.buf, Op.A, x)
  Ac_mul_B!(y, Op.A, Op.buf)
end

Ac_mul_B!(y, Op::AcA, x) = A_mul_B!(y, Op::AcA, x)

size(Op::AcA) = size(Op.A, 2), size(Op.A, 2)
eltype(Op::AcA) = eltype(Op.A)

# Shifted symmetric linear operator

struct Shift{M, T} <: LinOp
  A::M
  rho::T
  function Shift{M, T}(A::M, rho::T) where {M, T}
    if eltype(A) != T
      error("type of rho is incompatible with A")
    end
    new(A, rho)
  end
end

Shift(A::M, rho::T) where {M, T} = Shift{M, T}(A, rho)

function A_mul_B!(y, Op::Shift, x)
  A_mul_B!(y, Op.A, x)
  y .+= Op.rho .* x
end

function Ac_mul_B!(y, Op::Shift, x)
  Ac_mul_B!(y, Op.A, x)
  y .+= Op.rho .* x
end

size(Op::Shift) = size(Op.A, 2), size(Op.A, 2)
eltype(Op::Shift) = eltype(Op.A)
