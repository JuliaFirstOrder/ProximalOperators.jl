# (weighted) sum of L2 norm and L1 norm
# for Group Lasso, this can be used together with src/calculus/slicedSeparableSum
export NormL1plusL2

"""
    NormL1plusL2(λ_1=1, λ_2=1)

With two nonegative scalars λ_1 and λ_2, return the function
```math
f(x) = λ_1 ∑_{i=1}^{n} |x_i| + λ_2 \\sqrt{x_1^2 + … + x_n^2}.
```
With nonnegative array λ_1 and nonnegative scalar λ_2, return the function
```math
f(x) =  ∑_{i=1}^{n} {λ_1}_i |x_i| + λ_2 \\sqrt{x_1^2 + … + x_n^2}.
```
"""
struct NormL1plusL2{L1<:NormL1, L2 <: NormL2}
    l1::L1
    l2::L2
end

is_separable(f::Type{<:NormL1plusL2}) = false
is_convex(f::Type{<:NormL1plusL2}) = true
is_positively_homogeneous(f::Type{<:NormL1plusL2}) = true

function NormL1plusL2(lambda1::L=1, lambda2::M=1) where {L <: Union{Real, AbstractArray}, M <: Real}
    NormL1plusL2(NormL1(lambda1), NormL2(lambda2))
end

function (f::NormL1plusL2)(x::AbstractArray{T}) where T <: RealOrComplex
    return f.l1(x) + f.l2(x)
end

function prox!(y::AbstractArray{T}, f::NormL1plusL2, x::AbstractArray{T}, gamma::Real=1) where T <: RealOrComplex
    prox!(y, f.l1, x, gamma)
    prox!(y, f.l2, y, gamma)
    return f(y)
end

function prox_naive(f::NormL1plusL2, x::AbstractArray{T}, gamma::Real=1) where T <: RealOrComplex
    y1, v1 = prox_naive(f.l1, x, gamma)
    y2, v2 = prox_naive(f.l2, y1, gamma)
    return y2, f(y2)
end
