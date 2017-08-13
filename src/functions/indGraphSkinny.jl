# This implements the variant when A is dense and m > n

export IndGraphSkinny

struct IndGraphSkinny{T <: RealOrComplex} <: IndGraph
  m::Int
  n::Int
  A::Array{T,2}
  AA::Array{T,2}
  F::Base.LinAlg.Cholesky{T, Array{T, 2}} #LL factorization
  tmp::Array{T,1}
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
  IndGraphSkinny(m, n, A, AA, F, tmp)
end

function (f::IndGraphSkinny)(x::AbstractArray{T}, y::AbstractArray{T}) where
    {T <: RealOrComplex}
  # the tolerance in the following line should be customizable
  A_mul_B!(f.tmp, f.A, x)
  f.tmp .-= y
  if norm(f.tmp, Inf) <= 1e-10
    return 0.0
  end
  return +Inf
end

function prox!(
    x::AbstractArray{T},
    y::AbstractArray{T},
    f::IndGraphSkinny,
    c::AbstractArray{T},
    d::AbstractArray{T}
    ) where {T <: RealOrComplex}

  # x[:] = f.F \ (c + f.A' * d)
  At_mul_B!(x, f.A, d)
  x .+= c
  A_ldiv_B!(f.F, x)

  A_mul_B!(y, f.A, x)
  return 0.0
end

fun_name(f::IndGraphSkinny) = "Indicator of an operator graph defined by dense full column rank matrix"
# fun_dom(f::IndGraph) = "Array{Real,1}, Array{Complex,1}"
# fun_expr(f::IndGraph) = "x,y ↦ 0 if Ax = y, +∞ otherwise"
# fun_params(f::IndGraph) =
#   string( "A = ", typeof(f.A), " of size ", size(f.A))

function prox_naive(
    f::IndGraphSkinny,
    c::AbstractArray{T},
    d::AbstractArray{T}
    ) where {T <: RealOrComplex}

  x = f.F \ (c + f.A' * d)
  return x, f.A * x, 0.0
end

## Additional signatures
# prox!(xy::Tuple{AbstractVector{T},AbstractVector{T}},
#       f::IndGraphSkinny,
#       cd::Tuple{AbstractVector{T},AbstractVector{T}}
#   ) where {T <: RealOrComplex} =
#     prox!(xy[1], xy[2], f, cd[1], cd[2])
#
function (f::IndGraphSkinny)(xy::AbstractVector{T}) where
  {T <: RealOrComplex}
  x, y = splitinput(f, xy)
  return f(x, y)
end
