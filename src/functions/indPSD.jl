# indicator of a PSD

export IndPSD

"""
**Indicator of the set of positive semi-definite cone**

    IndPSD(;scaling=false)

Returns the indicator of the set
```math
C = \\{ X : X \\succeq 0 \\}.
```
The argument to the function can be either a `Symmetric` or `Hermitian` object,
or an object of type `AbstractVector{Float64}` holding a symmetric matrix in (lower triangular) packed storage.

If `scaling = true` then the vectors `y` and `x` in
`prox!(y::AbstractVector{Float64}, f::IndPSD, x::AbstractVector{Float64}, args...)`
have the off-diagonal elements multiplied with `√2` to preserve inner products,
see Vandenberghe 2010: http://www.seas.ucla.edu/~vandenbe/publications/coneprog.pdf .

I.e. when when `scaling=true`, let `X,Y` be matrices and

`x = (X_{1,1}, √2⋅X_{2,1}, ... ,√2⋅X_{n,1}, X_{2,2}, √2⋅X_{3,2}, ..., X_{n,n})`,

`y = (Y_{1,1}, √2⋅Y_{2,1}, ... ,√2⋅Y_{n,1}, Y_{2,2}, √2⋅Y_{3,2}, ..., Y_{n,n})`

then `prox!(Y, f, X)` is equivalent to `prox!(y, f, x)`.
"""
struct IndPSD <: ProximableFunction
    scaling::Bool
end

IndPSD(; scaling=false) = IndPSD(scaling)

function (f::IndPSD)(X::HermOrSym{T}) where {R <: Real, T <: RealOrComplex{R}}
    F = eigen(X)
    for i in eachindex(F.values)
        #Do we allow for some tolerance here?
        if F.values[i] <= -10 * eps(R)
            return R(Inf)
        end
    end
    return R(0)
end

is_convex(f::IndPSD) = true
is_cone(f::IndPSD) = true

function prox!(Y::HermOrSym{T}, f::IndPSD, X::HermOrSym{T}, gamma::Real=1.0) where {R <: Real, T <: RealOrComplex{R}}
    n = size(X, 1)
    F = eigen(X)
    for i in eachindex(F.values)
        F.values[i] = max.(R(0), F.values[i])
    end
    for i = 1:n
        for j = 1:n
            Y.data[i, j] = R(0)
            for k = 1:n
                Y.data[i, j] += F.vectors[i, k] * F.values[k] * conj(F.vectors[j, k])
            end
        end
    end
    return R(0)
end

fun_name(f::IndPSD) = "indicator of positive semidefinite cone"
fun_dom(f::IndPSD) = "Symmetric, Hermitian, AbstractArray{Float64}"
fun_expr(f::IndPSD) = "x ↦ 0 if A ⪰ 0, +∞ otherwise"
fun_params(f::IndPSD) = "none"

function prox_naive(f::IndPSD, X::HermOrSym{T}, gamma::Real=1.0) where {R <: Real, T <: RealOrComplex{R}}
    F = eigen(X)
    return F.vectors * Diagonal(max.(R(0), F.values)) * F.vectors', R(0)
end

"""
Scales the diagonal of `x` with `val`, where `x` is the lower triangualar part
of a matrix, stored column by column.
"""
function scale_diagonal!(x, val)
    n = Int(sqrt(1/4+2*length(x))-1/2)
    k = -n
    for i = 1:n
        k += n - i + 2        #Calculate indices of diagonal elements recursively (paralell faster?)
        x[k] *= val            #Scale diagonal
    end
end

## Below: with AbstractVector argument

function (f::IndPSD)(x::AbstractVector{T}) where T <: Float64
    y = copy(x)
    # If scaling, scale diagonal (eigenvalues scaled by sqrt(2))
    f.scaling && scale_diagonal!(y, sqrt(2))

    Z = dspev!(:N, :L, y)
    for i in 1:length(Z)
        # Do we allow for some tolerance here?
        if Z[i] <= -1e-14
            return +Inf
        end
    end
    return 0.0
end

function prox!(y::AbstractVector{Float64}, f::IndPSD, x::AbstractVector{Float64}, gamma::Real=1.0)
    # Copy x since dspev! corrupts input
    y .= x                            

    # If scaling, scale diagonal
    f.scaling && scale_diagonal!(y, sqrt(2))

    (W, Z) = dspevV!(:L, y)
    # NonNeg eigenvalues
    W = max.(W, 0.0)
    # Equivalent to Z*diagm(W) without constructing W matrix
    M = Z.*W'
    # Now let M = Z*diagm(W)*Z'
    M = M*Z'
    n = length(W)
    k = 1
    # Store lower diagonal of M in y
    for j in 1:n, i in j:n
        y[k] = M[i,j]
        k = k+1
    end

    # If scaling, un-scale diagonal
    f.scaling && scale_diagonal!(y, 1/sqrt(2))

    return 0.0
end

function prox_naive(f::IndPSD, x::AbstractVector{T}, gamma::Real=1.0) where T<:Float64
    # Formula for size of matrix
    n = Int(sqrt(1/4+2*length(x))-1/2)
    X = Array{T,2}(undef, n, n)
    k = 1
    # Store y in M
    for j = 1:n, i = j:n
        # Lower half
        X[i,j] = x[k]
        if i != j
            # Strictly upper half
            X[j,i] = x[k]
        end
        k = k+1
    end
    # Scale diagonal elements by sqrt(2)
    # See Vandenberghe 2010 http://www.seas.ucla.edu/~vandenbe/publications/coneprog.pdf
    # It's equivalent to scaling off-diagonal by 1/sqrt(2) and working with sqrt(2)*X
    if f.scaling
        for i = 1:n
            X[i,i] *= sqrt(2)
        end
    end
    X, v = prox_naive(f, Symmetric(X), gamma)

    # Scale diagonal elements back
    if f.scaling
        for i = 1:n
            X[i,i] /= sqrt(2)
        end
    end

    y = similar(x)
    k = 1
    # Store Lower half of X in y
    for j = 1:n, i = j:n
        y[k] = X[i,j]
        k = k+1
    end
    return y, 0.0
end

## Below: with AbstractMatrix argument (wrap in Symmetric or Hermitian)

function (f::IndPSD)(X::AbstractMatrix{R}) where R <: Real
    f(Symmetric(X))
end
    
function prox!(y::AbstractMatrix{R}, f::IndPSD, x::AbstractMatrix{R}, gamma::Real=1.0) where R <: Real
    prox!(Symmetric(y), f, Symmetric(x), gamma)
end

function prox_naive(f::IndPSD, X::AbstractMatrix{R}, gamma::Real=1.0) where R <: Real
    prox_naive(f, Symmetric(X), gamma)
end

function (f::IndPSD)(X::AbstractMatrix{C}) where C <: Complex
    f(Hermitian(X))
end
    
function prox!(y::AbstractMatrix{C}, f::IndPSD, x::AbstractMatrix{C}, gamma::Real=1.0) where C <: Complex
    prox!(Hermitian(y), f, Hermitian(x), gamma)
end

function prox_naive(f::IndPSD, X::AbstractMatrix{C}, gamma::Real=1.0) where C <: Complex
    prox_naive(f, Hermitian(X), gamma)
end
