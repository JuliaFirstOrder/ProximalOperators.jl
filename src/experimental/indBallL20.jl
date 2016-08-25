# indicator function of the matrices with at most

"""
  IndBallL20(r::Int64, dim=1)

For an integer parameter `r > 0`, if dim=1 then returns the function
`g = ind{X : countnz(||X(:,i)||_2) ⩽ r}`, if dim=2 then instead
`g = ind{X : countnz(||X(i,:)||_2) ⩽ r}`.
"""

immutable IndBallL20 <: IndicatorFunction
  r::Int64
  dim::Int
  IndBallL20(r::Int64, dim=1) =
    r <= 0 ? error("parameter r must be a positive integer") : new(r, dim)
end

function call(f::IndBallL20, X::RealOrComplexArray)
  if countnz(sqrt(sum(abs(X).^2,dim))) > f.r return +Inf end
  return 0.0
end

function prox(f::IndBallL20, X::RealOrComplexArray, gamma::Float64=1.0)
  Y = zeros(X)
  if f.r < log2(size(X,dim))
    p = selectperm(sqrt(sum(abs(X).^2,dim)[:]), 1:f.r, rev=true)
    if dim == 1
    Y[:,p] = X[:,p]
    elseif dim == 2
    Y[p,:] = X[p,:]
    end
  else
    p = sortperm(sqrt(sum(abs(X).^2,dim)[:]), rev=true)
    if dim == 1
    Y[:,p[1:r]] = X[:,p[1:r]]
    elseif dim == 2
    Y[p[1:r],:] = X[p[1:r],:]
    end
  end
  return Y, 0.0
end
