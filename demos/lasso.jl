# lasso.jl - Lasso solvers based on FISTA and ADMM using ProximalOperators

using ProximalOperators

# Define solvers

function lasso_fista(A, b, lam, x; tol=1e-4, maxit=50000)
  x_prev = copy(x)
  g = NormL1(lam)
  gam = 1.0/norm(A)^2
  for it = 1:maxit
    x_extr = x + (it-2)/(it+1)*(x - x_prev)
    res = A*x_extr - b
    y = x_extr - gam*(A'*res)
    x_prev .= x
    prox!(x, g, y, gam)
    if norm(x_extr-x, Inf)/gam <= tol*(1+norm(x, Inf))
      break
    end
  end
  return x
end

function lasso_admm(A, b, lam, x; tol=1e-4, maxit=50000)
  u = zeros(x)
  z = copy(x)
  f = LeastSquares(A, b)
  g = NormL1(lam)
  gam = 10.0/norm(A)^2
  for it = 1:maxit
    prox!(x, f, z - u, gam)
    prox!(z, g, x + u, gam)
    if norm(x-z, Inf) <= tol*(1+norm(u, Inf))
      break
    end
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
println("  nnz(x)    = $(norm(x_fista, 0))")
println("  obj value = $(0.5*norm(A*x_fista-b)^2 + lam*norm(x_fista, 1))")

x_admm = lasso_admm(A, b, lam, zeros(n))
println("ADMM")
println("  nnz(x)    = $(norm(x_admm, 0))")
println("  obj value = $(0.5*norm(A*x_admm-b)^2 + lam*norm(x_admm, 1))")
