# This implements the variant when A is dense and m > n

struct IndGraphSkinny{T <: RealOrComplex} <: IndGraph
  m::Int
  n::Int
  A::Array{T,2}
  AA::Array{T,2}
  F::Base.LinAlg.Cholesky{T, Array{T, 2}} #LL factorization
  tmp::Array{T,1}
  tmpx::SubArray{T, 1, Array{T, 1}, Tuple{UnitRange{Int64}}, true}
end

function IndGraphSkinny(A::Array{T,2}) where {T <: RealOrComplex}
  m, n = size(A)
  AA = A' * A

  F = LinAlg.cholfact(AA + eye(n))
  #normrows = vec(sqrt.(sum(abs2.(A), 2)))

  #The tmp vector assumes that difference between m and n is not drastic.
  # If someone will decide to solve it using m >> n, then tmp vector might be
  # considered to have only n position required to prox esitmation and indicator
  # calculation might be converted to less efficient.
  tmp = Array{T, 1}(m)
  IndGraphSkinny(m, n, A, AA, F, tmp, view(tmp, 1:n))
end

function (f::IndGraphSkinny){T <: RealOrComplex}(x::Array{T,1}, y::Array{T, 1})
  # the tolerance in the following line should be customizable
  A_mul_B!(f.tmp, f.A, x)
  f.tmp .-= y
  if norm(f.tmp, Inf) <= 1e-10
    return 0.0
  end
  return +Inf
end

function prox!{T <: RealOrComplex}(
    x::Array{T, 1},
    y::Array{T, 1},
    f::IndGraphSkinny,
    c::Array{T, 1},
    d::Array{T, 1})

  # x[:] = f.F \ (c + f.A' * d)
  At_mul_B!(f.tmpx, f.A, d)
  f.tmpx .+= c
  A_ldiv_B!(x, f.F, f.tmpx)

  A_mul_B!(y, f.A, x)
  return 0.0
end

fun_name(f::IndGraphSkinny) = "Indicator of an operator graph defined by dense full column rank matrix"
# fun_dom(f::IndGraph) = "Array{Real,1}, Array{Complex,1}"
# fun_expr(f::IndGraph) = "x,y ↦ 0 if Ax = y, +∞ otherwise"
# fun_params(f::IndGraph) =
#   string( "A = ", typeof(f.A), " of size ", size(f.A))

function prox_naive{T <: RealOrComplex}(
    f::IndGraphSkinny,
    c::Array{T, 1},
    d::Array{T, 1})

  x = f.F \ (c + f.A' * d)
  return x, f.A * x, 0.0
end
