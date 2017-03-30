# Precompose with a Gram-diagonal linear mapping and translation

"""
  PrecomposeGramDiagonal(f::ProximableFunction, L::Function, Ladj::Function, mu, b)

Returns the function `g(x) = f(L(x) + b)`. L is a linear mapping, and Ladj is its adjoint. Mapping L must be Gram-diagonal, i.e., `L(Ladj(y)) = μ.*y` for `μ ⩾ 0`. Furthermore, either `f` is separable or `μ` is constant (or scalar), for the `prox` of `g` to be computable.
"""

immutable PrecomposeGramDiagonal{T <: ProximableFunction, R <: Union{Real, AbstractArray}, S <: Union{Real, AbstractArray}} <: ProximableFunction
  f::T
  L::Function
  Ladj::Function
  mu::R
  b::S
  function PrecomposeGramDiagonal(f::T, L::Function, Ladj::Function, mu::R, b::S)
    if !(eltype(a) <: Real)
      error("a must have real elements")
    end
    muconst = (maximum(mu) == minimum(mu))
    if ~muconst || ~is_separable(f)
      error("either f is separable or μ is constant")
    end
    if any(mu <= 0.0)
      error("elements of μ must be positive")
    else
      new(f, L, Ladj, mu, b)
    end
  end
end

is_prox_accurate(f::PrecomposeGramDiagonal) = is_prox_accurate(f.f)

PrecomposeGramDiagonal{T <: ProximableFunction, R <: Union{Real, AbstractArray}, S <: Real}(f::T, L, Ladj, mu::R, b::S=zero(S)) = PrecomposeGramDiagonal{T, R, S}(f, L, Ladj, mu, b)

PrecomposeGramDiagonal{T <: ProximableFunction, R <: Union{Real, AbstractArray}, S <: AbstractArray}(f::T, L, Ladj, mu::R, b::S) = PrecomposeGramDiagonal{T, R, S}(f, L, Ladj, mu, b)

function (g::PrecomposeGramDiagonal){T <: RealOrComplex}(x::AbstractArray{T})
  return g.f(g.L(x) .+ g.b)
end

function prox!{T <: RealOrComplex}(y::AbstractArray{T}, g::PrecomposeGramDiagonal, x::AbstractArray{T}, gamma::Real=1.0)
  # TODO: implement this

end

function prox_naive{T <: RealOrComplex}(g::PrecomposeGramDiagonal, x::AbstractArray{T}, gamma::Real=1.0)
  # TODO: implement this

end

fun_name(f::PrecomposeGramDiagonal) = string("Precomposition by affine Gram-diagonal mapping of ", fun_name(f.f))
fun_dom(f::PrecomposeGramDiagonal) = fun_dom(f.f)
fun_expr(f::PrecomposeGramDiagonal) = "x ↦ f(L(x) + b)"
