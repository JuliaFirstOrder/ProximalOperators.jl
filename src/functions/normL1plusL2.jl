# (weighted) sum of L2 norm and L1 norm
# for Group Lasso, this can be used together with src/calculus/slicedSeparableSum
export NormL1plusL2

"""
**L1 norm plus L2 norm**

    NormL1plusL2(λ_1=1, λ_2=1)

With two nonnegative scalar parameters λ_1 and λ_2, returns the function
```math
f(x) = λ_1 ∑_{i=1}^{n} |x_i| + λ_2 \\sqrt{x_1^2 + … + x_n^2}.
```
"""
struct NormL1plusL2{L <: Real, M <: Real} <: ProximableFunction
    lambda1::L
    lambda2::M
    function NormL1plusL2{L, M}(lambda1::L, lambda2::M) where {L <: Real, M <: Real}
        if lambda1 < 0 || lambda2 < 0
            error("parameters λ_1, λ_2 must be nonnegative")
        else
            new(lambda1, lambda2)
        end
    end
end

is_separable(f::NormL1plusL2) = false
is_convex(f::NormL1plusL2) = true
is_positively_homogeneous(f::NormL1plusL2) = true

NormL1plusL2(lambda1::L=1, lambda2::M=1) where {L <: Real, M <: Real} = NormL1plusL2{L, M}(lambda1, lambda2)

function (f::NormL1plusL2)(x::AbstractArray)
    return f.lambda1 * norm(x, 1) + f.lambda2 * norm(x, 2)
end

function prox!(y::AbstractArray{T}, f::NormL1plusL2, x::AbstractArray{T}, gamma::Real=1) where T <: Real

    f1 = NormL1(f.lambda1)
    f2 = NormL2(f.lambda2)

    prox!(y, f1, x, gamma)
    prox!(y, f2, y, gamma)

    return f.lambda1 * norm(y, 1) + f.lambda2 * norm(y, 2)

end

fun_name(f::NormL1plusL2) = "L2-norm + L1-norm"
fun_dom(f::NormL1plusL2) = "AbstractArray{Real}"
fun_expr(f::NormL1plusL2) = "x ↦ λ_1 ||x||_1 + λ_2||x||_2"
fun_params(f::NormL1plusL2) = "λ_1 = $(f.lambda1), λ_2 = $(f.lambda2)"
