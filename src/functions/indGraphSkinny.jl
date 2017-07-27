# This implements the variant when A is dense and m > n

struct IndGraphSkinny{T <: RealOrComplex} <: IndGraph
  m::Int
  n::Int
  A::AbstractArray{T,2}
  AA::AbstractArray{T,2}
  F::Base.LinAlg.Cholesky #LDL factorization
end

function IndGraphSkinny(A::AbstractArray{T,2}) where {T <: RealOrComplex}
  m, n = size(A)
  AA = A' * A

  F = LinAlg.cholfact(AA + speye(n))
  #normrows = vec(sqrt.(sum(abs2.(A), 2)))

  IndGraphSkinny(m, n, A, AA, F)
end

function (f::IndGraphSkinny){T <: RealOrComplex}(x::AbstractArray{T,1}, y::AbstractArray{T, 1})
  # the tolerance in the following line should be customizable
  if norm(f.A * x - y, Inf) <= 1e-10
    return 0.0
  end
  return +Inf
end

function prox!{T <: RealOrComplex}(
    x::AbstractArray{T, 1},
    y::AbstractArray{T, 1},
    f::IndGraphSkinny,
    c::AbstractArray{T, 1},
    d::AbstractArray{T, 1})

  x[:] = f.F \ (c + f.A' * d)
  A_mul_B!(y, f.A, x)
  return 0.0
end

fun_name(f::IndGraphSkinny) = "Indicator of an operator graph defined by dense full column rank matrix"
# fun_dom(f::IndGraph) = "AbstractArray{Real,1}, AbstractArray{Complex,1}"
# fun_expr(f::IndGraph) = "x,y ↦ 0 if Ax = y, +∞ otherwise"
# fun_params(f::IndGraph) =
#   string( "A = ", typeof(f.A), " of size ", size(f.A))

function prox_naive{T <: RealOrComplex}(
    f::IndGraphSkinny,
    c::AbstractArray{T, 1},
    d::AbstractArray{T, 1})

  x = f.F \ (c + f.A' * d)
  return x, f.A * x, 0.0
end
