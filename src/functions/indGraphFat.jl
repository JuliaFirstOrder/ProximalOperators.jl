# This implements the variant when A is dense and m < n

struct IndGraphFat{T <: RealOrComplex} <: IndGraph
  m::Int
  n::Int
  A::AbstractArray{T,2}
  AA::AbstractArray{T,2}
  F::Base.LinAlg.Cholesky #LDL factorization
end

function IndGraphFat(A::AbstractArray{T,2}) where {T <: RealOrComplex}
  m, n = size(A)
  AA = A * A'

  F = LinAlg.cholfact(eye(m) + AA)
  #normrows = vec(sqrt.(sum(abs2.(A), 2)))

  IndGraphFat(m, n, A, AA, F)
end

# is_convex(f::IndGraph) = true
# is_set(f::IndGraph) = true
# is_cone(f::IndGraph) = false

function prox!{T <: RealOrComplex}(
    x::AbstractArray{T, 1},
    y::AbstractArray{T, 1},
    f::IndGraphFat,
    c::AbstractArray{T, 1},
    d::AbstractArray{T, 1})
  y[:] = f.F \ (f.A * c + f.AA * d)
  At_mul_B!(x, f.A, d - y)
  x[:] += c
  return 0.0
end

function (f::IndGraphFat){T <: RealOrComplex}(x::AbstractArray{T,1}, y::AbstractArray{T, 1})
  # the tolerance in the following line should be customizable
  if norm(f.A * x - y, Inf) <= 1e-10
    return 0.0
  end
  return +Inf
end

fun_name(f::IndGraphFat) = "Indicator of an operator graph defined by dense full row rank matrix"
# fun_dom(f::IndGraph) = "AbstractArray{Real,1}, AbstractArray{Complex,1}"
# fun_expr(f::IndGraph) = "x,y ↦ 0 if Ax = y, +∞ otherwise"
# fun_params(f::IndGraph) =
#   string( "A = ", typeof(f.A), " of size ", size(f.A))

function prox_naive{T <: RealOrComplex}(
    f::IndGraphFat,
    c::AbstractArray{T, 1},
    d::AbstractArray{T, 1})

  y = f.F \ (f.A * c + f.AA * d)
  return c + f.A' * (d - y), y, 0.0
end
