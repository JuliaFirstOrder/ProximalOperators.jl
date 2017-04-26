# lasso.jl - Lasso solvers based on FISTA and ADMM using ProximalOperators
#
#   minimize 0.5*||A*x - b||^2 + lam*||x||_1
#

using ProximalOperators

# Define solvers

function lasso_fista(A, b, lam, x; tol=1e-3, maxit=50000)
  x_prev = copy(x)
  g = NormL1(lam)
  gam = 1.0/norm(A)^2
  for it = 1:maxit
    # extrapolation step
    x_extr = x + (it-2)/(it+1)*(x - x_prev)
    # compute least-squares residual
    res = A*x_extr - b
    # compute gradient (forward) step
    y = x_extr - gam*(A'*res)
    # store current iterate
    x_prev .= x
    # compute proximal (backward) step
    prox!(x, g, y, gam)
    # stopping criterion
    if norm(x_extr-x, Inf)/gam <= tol*(1+norm(x, Inf))
      break
    end
  end
  return x
end

function lasso_admm(A, b, lam, x; tol=1e-5, maxit=50000)
  u = zeros(x)
  z = copy(x)
  f = LeastSquares(A, b)
  g = NormL1(lam)
  gam = 10.0/norm(A)^2
  for it = 1:maxit
    # perform f-update step
    prox!(x, f, z - u, gam)
    # perform g-update step
    prox!(z, g, x + u, gam)
    # stopping criterion
    if norm(x-z, Inf) <= tol*(1+norm(u, Inf))
      break
    end
    # dual update
    u .+= x - z
  end
  return z
end

# Generate random problem

println("Generating random lasso problem")

m, n, k, sig = 500, 2500, 100, 1e-3
A = randn(m, n)
x_true = [randn(k)..., zeros(n-k)...]
b = A*x_true + sig*randn(m)
lam = 0.1*norm(A'*b, Inf)

# Call solvers

println("Calling solvers")

x_fista = lasso_fista(A, b, lam, zeros(n))
println("FISTA")
println("      nnz(x)    = $(norm(x_fista, 0))")
println("      obj value = $(0.5*norm(A*x_fista-b)^2 + lam*norm(x_fista, 1))")

x_admm = lasso_admm(A, b, lam, zeros(n))
println("ADMM")
println("      nnz(x)    = $(norm(x_admm, 0))")
println("      obj value = $(0.5*norm(A*x_admm-b)^2 + lam*norm(x_admm, 1))")
