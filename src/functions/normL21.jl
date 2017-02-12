# L2,1 norm/Sum of norms of columns or rows (times a constant)

"""
  NormL21(λ::Real=1.0, dim=1)

Returns the function `g(X) = λsum(||x_i||)` for a nonnegative parameter `λ ⩾ 0`,
where x_i are the columns of X if dim == 1, or the rows of X if dim == 2.
In words, it is the (weighted) sum of the Euclidean norm of the columns (rows) of X.
"""

immutable NormL21{R <: Real, I <: Integer} <: ProximableConvex
  lambda::R
  dim::I
  function NormL21(lambda::R, dim::I)
    if lambda < 0
      error("parameter λ must be nonnegative")
    else
      new(lambda, dim)
    end
  end
end

NormL21{R <: Real, I <: Integer}(lambda::R=1.0, dim::I=1) = NormL21{R, I}(lambda, dim)

function (f::NormL21){T <: RealOrComplex}(X::AbstractArray{T,2})
  nslice = 0.0
  n21X = 0.0
  if f.dim == 1
    for j = 1:size(X,2)
      nslice = 0.0
      for i = 1:size(X,1)
      	nslice += abs(X[i,j])^2
      end
      n21X += sqrt(nslice)
    end
  elseif f.dim == 2
    for i = 1:size(X,1)
      nslice = 0.0
      for j = 1:size(X,2)
      	nslice += abs(X[i,j])^2
      end
      n21X += sqrt(nslice)
    end
  end
  return f.lambda*n21X
end

function prox!{T <: RealOrComplex}(f::NormL21, X::AbstractArray{T,2}, Y::AbstractArray{T,2}, gamma::Real=1.0)
  gl = gamma*f.lambda
  nslice = zero(Float64)
  n21X = zero(Float64)
  if f.dim == 1
    for j = 1:size(X,2)
      nslice = 0.0
      for i = 1:size(X,1)
      	nslice += abs(X[i,j])^2
      end
      nslice = sqrt(nslice)
      scal = 1-gl/nslice
      scal = scal <= 0.0 ? 0.0 : scal
      for i = 1:size(X,1)
        Y[i,j] = scal*X[i,j]
      end
      n21X += scal*nslice
    end
  elseif f.dim == 2
    for i = 1:size(X,1)
      nslice = 0.0
      for j = 1:size(X,2)
      	nslice += abs(X[i,j])^2
      end
      nslice = sqrt(nslice)
      scal = 1-gl/nslice
      scal = scal <= 0.0 ? 0.0 : scal
      for j = 1:size(X,2)
        Y[i,j] = scal*X[i,j]
      end
      n21X += scal*nslice
    end
  end
  return f.lambda*n21X
end

fun_name(f::NormL21) = "sum of Euclidean norms"
fun_dom(f::NormL21) = "AbstractArray{Real,2}, AbstractArray{Complex,2}"
fun_expr(f::NormL21) = "x ↦ λsum(||x_i||)"
fun_params(f::NormL21) = "λ = $(f.lambda), dim = $(f.dim)"

function prox_naive{T <: RealOrComplex}(f::NormL21, X::AbstractArray{T,2}, gamma::Real=1.0)
  Y = max.(0, 1-f.lambda*gamma./sqrt.(sum(abs.(X).^2, f.dim))).*X
  return Y, f.lambda*sum(sqrt.(sum(abs.(Y).^2, f.dim)))
end
