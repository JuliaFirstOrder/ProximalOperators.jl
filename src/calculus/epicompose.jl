# TODO *Work in progress*, this is highly incomplete code

# Epi-composition, also known as infimal postcomposition.
# This is the dual operation to precomposition, see Rockafellar and Wets,
# "Variational Analisys", Theorem 11.23.
#
# Given a function f and a linear operator L, their epi-composition is:
#
#   g(y) = (Lf)(y) = inf_x { f(x) : Lx = y }.
#
# Plugging g directly in the definition of prox, one has:
#
#   prox_{\gamma g}(z) = argmin_y { (Lf)(y) + 1/(2\gamma)||y - z||^2 }
#     = argmin_y { inf_x { f(x) : Lx = y } + 1/(2\gamma)||y - z||^2 }
#     = L * argmin_x { f(x) + 1/(2\gamma)||Lx - z||^2 }.
#
# When L is such that L'*L = mu*Id, then this just requires prox_{\gamma f}.
# In some other cases the prox can be "easily" computed, such as when f is
# quadratic or extended-quadratic.

export Epicompose

type Epicompose{S <: AbstractMatrix, T <: ProximableFunction, F <: Factorization} <: ProximableFunction
  L::S
  f::T
  iter::Bool
  mu
  gamma
  fact::F
  function Epicompose{S, T, F}(L::S, f::T, iter::Bool) where {T <: ProximableFunction, S <: AbstractMatrix, F <: Factorization}
    new(L, f, iter, 0.0, -1.0)
  end
end

function Epicompose(L::S, f::T, iter::Bool=false) where {T <: ProximableFunction, S <: AbstractMatrix}
  # TODO this is incomplete
  Epicompose{S, T, Factorization}(L, f, iter)
end

function prox!(y, g::Epicompose{S, T}, x, gamma) where {T <: ProximableFunction, S <: AbstractMatrix}
  # this line here allocates stuff
  z = (g.L'*x)/f.mu
  p, v = prox(g.f, z, gamma/f.mu)
  A_mul_B!(y, f.L, p)
  return v
end

function prox!(y, g::Epicompose{S, T}, x, gamma) where {T <: Quadratic, S <: AbstractMatrix}
  if g.iter == false
    if gamma != g.gamma
      factor_step!(g, gamma)
    end
    # this line here allocates stuff
    p = g.fact\((g.L'*x)/gamma - g.f.q)
    fy = 0.5*vecdot(p, g.f.Q*p) + vecdot(p, g.f.q)
  else
    error("not implemented")
  end
  A_mul_B!(y, g.L, p)
  return fy
end

function factor_step!(g::Epicompose{S, T}, gamma::R) where {T <: Quadratic, S <: AbstractMatrix, R <: Real}
  g.gamma = gamma;
  g.fact = cholfact(g.f.Q + (g.L'*g.L)/gamma);
end

fun_expr(g::Epicompose) = "x ↦ inf { f(y) : Ly = x }"
