# This implements the variant when A is dense and m < n

struct IndGraphFat{T <: RealOrComplex} <: IndGraph
  m::Int
  n::Int
  A::Array{T,2}
  AA::Array{T,2}
  tmp::Array{T,1}
  # tmpx::Array{T,1}
  F::Base.LinAlg.Cholesky{T, Array{T, 2}} #LL factorization
end

function IndGraphFat(A::Array{T,2}) where {T <: RealOrComplex}
  m, n = size(A)
  AA = A * A'

  F = LinAlg.cholfact(eye(m) + AA)
  #normrows = vec(sqrt.(sum(abs2.(A), 2)))

  IndGraphFat(m, n, A, AA, Array{T, 1}(m), F)
end

# is_convex(f::IndGraph) = true
# is_set(f::IndGraph) = true
# is_cone(f::IndGraph) = false

function prox!{T <: RealOrComplex}(
    x::Array{T, 1},
    y::Array{T, 1},
    f::IndGraphFat,
    c::Array{T, 1},
    d::Array{T, 1})

  # y .= f.F \ (f.A * c + f.AA * d)
  A_mul_B!(f.tmp, f.A, c)
  A_mul_B!(y, f.AA, d)
  f.tmp .+= y
  A_ldiv_B!(y, f.F, f.tmp)

  # f.A * (d - y) + x
  copy!(f.tmp, d)
  f.tmp .-= y
  At_mul_B!(x, f.A, f.tmp)
  x .+= c
  return 0.0
end

function (f::IndGraphFat){T <: RealOrComplex}(x::Array{T,1}, y::Array{T, 1})
  # the tolerance in the following line should be customizable
  A_mul_B!(f.tmp, f.A, x)
  f.tmp .-= y
  if norm(f.tmp, Inf) <= 1e-10
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
    c::Array{T, 1},
    d::Array{T, 1})

  y = f.F \ (f.A * c + f.AA * d)
  return c + f.A' * (d - y), y, 0.0
end
