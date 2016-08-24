# L2,1 norm/Sum of norms of columns or rows (times a constant)

"""
  NormL21(λ::Float64=1.0, dim=1)

Returns the function `g(X) = λsum(||x_i||)` for a nonnegative parameter `λ ⩾ 0`,
where x_i are the columsn of X if dim == 1, or the rows of X if dim == 2.
In words, it is the (weighted) sum of the Euclidean norm of the columns (rows) of X.
"""

immutable NormL21 <: NormFunction
  lambda::Float64
  dim::Int
  NormL21(lambda::Float64=1.0, dim=1) =
    lambda < 0 ? error("parameter λ must be nonnegative") : new(lambda, dim)
end

function call(f::NormL21, X::RealOrComplexArray)
  return f.lambda*sum(sqrt(sum(abs(X).^2, f.dim)))
end

function prox(f::NormL21, gamma::Float64, X::RealOrComplexArray)
  Y = max(0, 1-f.lambda*gamma./sqrt(sum(abs(X).^2, f.dim))).*X
  return Y, f.lambda*sum(sqrt(sum(abs(Y).^2, f.dim)))
end
