# L2,1 norm/Sum of norms of columns or rows (times a constant)

export NormL21

"""
**Sum-of-``L_2`` norms**

    NormL21(λ=1, dim=1)

Returns the function
```math
f(X) = λ⋅∑_i\\|x_i\\|
```
for a nonnegative `λ`, where ``x_i`` is the ``i``-th column of ``X`` if `dim == 1`, and the ``i``-th row of ``X`` if `dim == 2`.
In words, it is the sum of the Euclidean norms of the columns or rows.
"""
struct NormL21{R <: Real, I <: Integer}
    lambda::R
    dim::I
    function NormL21{R,I}(lambda::R, dim::I) where {R <: Real, I <: Integer}
        if lambda < 0
            error("parameter λ must be nonnegative")
        else
            new(lambda, dim)
        end
    end
end

is_convex(f::NormL21) = true

NormL21(lambda::R=1, dim::I=1) where {R <: Real, I <: Integer} = NormL21{R, I}(lambda, dim)

function (f::NormL21)(X::AbstractArray{T, 2}) where {R, T <: RealOrComplex{R}}
    nslice = R(0)
    n21X = R(0)
    if f.dim == 1
        for j = 1:size(X, 2)
            nslice = R(0)
            for i = 1:size(X, 1)
                nslice += abs(X[i, j])^2
            end
            n21X += sqrt(nslice)
        end
    elseif f.dim == 2
        for i = 1:size(X, 1)
            nslice = R(0)
            for j = 1:size(X, 2)
                nslice += abs(X[i, j])^2
            end
            n21X += sqrt(nslice)
        end
    end
    return f.lambda * n21X
end

function prox!(Y::AbstractArray{T, 2}, f::NormL21, X::AbstractArray{T, 2}, gamma::Real=1) where {R, T <: RealOrComplex{R}}
    gl = gamma * f.lambda
    nslice = R(0)
    n21X = R(0)
    if f.dim == 1
        for j = 1:size(X, 2)
            nslice = R(0)
            for i = 1:size(X, 1)
                nslice += abs(X[i, j])^2
            end
            nslice = sqrt(nslice)
            scal = 1 - gl / nslice
            scal = scal <= 0 ? R(0) : scal
            for i = 1:size(X, 1)
                Y[i, j] = scal * X[i, j]
            end
            n21X += scal * nslice
        end
    elseif f.dim == 2
        for i = 1:size(X, 1)
            nslice = R(0)
            for j = 1:size(X, 2)
                nslice += abs(X[i, j])^2
            end
            nslice = sqrt(nslice)
            scal = 1-gl/nslice
            scal = scal <= 0 ? R(0) : scal
            for j = 1:size(X, 2)
                Y[i, j] = scal * X[i, j]
            end
            n21X += scal * nslice
        end
    end
    return f.lambda * n21X
end

function prox_naive(f::NormL21, X::AbstractArray{T,2}, gamma::Real=1.0) where T <: RealOrComplex
    Y = max.(0, 1 .- f.lambda * gamma ./ sqrt.(sum(abs.(X).^2, dims=f.dim))) .* X
    return Y, f.lambda * sum(sqrt.(sum(abs.(Y).^2, dims=f.dim)))
end
