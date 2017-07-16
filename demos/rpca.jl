# rpca.jl - Robust PCA solvers using ProximalOperators
#
#   minimize 0.5*||A - S - L||^2_F + lam1*||S||_1 + lam2*||L||_*
#
# See Parikh, Boyd "Proximal Algorithms", §7.2

using ProximalOperators

# Define solvers

function rpca_fista(A, lam1, lam2, S, L; tol=1e-3, maxit=50000)
  S_prev = copy(S)
  L_prev = copy(L)
  g = SeparableSum(NormL1(lam1), NuclearNorm(lam2))
  gam = 0.5
  for it = 1:maxit
    # extrapolation step
    S_extr = S + (it-2)/(it+1)*(S - S_prev)
    L_extr = L + (it-2)/(it+1)*(L - L_prev)
    # compute residual
    res = A - S - L
    # compute gradient (forward) step
    y_S = S_extr + gam*res
    y_L = L_extr + gam*res
    # store current iterates
    S_prev .= S
    L_prev .= L
    # compute proximal (backward) step
    prox!((S, L), g, (y_S, y_L), gam)
    # stopping criterion
    fix_point_res = max(vecnorm(S_extr-S, Inf), vecnorm(L_extr-L, Inf))/gam
    rel_fix_point_res = fix_point_res/(1+max(vecnorm(S,Inf), vecnorm(L,Inf)))
    if rel_fix_point_res <= tol
      break
    end
  end
  return S, L
end

# Generate random problem

println("Generating random robust PCA problem")

m, n, r, p, sig = 200, 500, 4, 0.05, 1e-3
L1 = randn(m, r)
L2 = randn(r, n)
L = L1*L2
S = sprand(m, n, p)
S = 20*S - 10.*S
V = sig*randn(m, n)
A = L + S + V
lam1 = 0.15*vecnorm(A, Inf)
lam2 = 0.15*norm(A)

# Call solvers

println("Calling solvers")

S_fista, L_fista = rpca_fista(A, lam1, lam2, zeros(m, n), zeros(m, n))
println("FISTA")
println("      nnz(S)    = $(vecnorm(S_fista, 0))")
println("      rank(L)   = $(rank(L_fista))")
println("      ||A||     = $(vecnorm(A, Inf))")
println("      ||A-S-L|| = $(vecnorm(A - S_fista - L_fista, Inf))")
