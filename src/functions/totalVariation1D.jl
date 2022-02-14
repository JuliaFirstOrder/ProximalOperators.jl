# 1-dimensional Total Variation (times a constant)

export TotalVariation1D

"""
    TotalVariation1D(λ=1)

With a nonnegative scalar parameter λ, return the 1D total variation
```math
f(x) = λ ∑_{i=2}^{n} |x_i - x_{i-1}|.
```
"""
struct TotalVariation1D{T}
    lambda::T
    function TotalVariation1D{T}(lambda::T) where T
        if lambda < 0
            error("parameter λ must be nonnegative")
        else
            new(lambda)
        end
    end
end

is_separable(f::Type{<:TotalVariation1D}) = false
is_convex(f::Type{<:TotalVariation1D}) = true
is_positively_homogeneous(f::Type{<:TotalVariation1D}) = true

TotalVariation1D(lambda::R=1) where R = TotalVariation1D{R}(lambda)

function (f::TotalVariation1D)(x)
    return f.lambda * norm(x[2:end] - x[1:end-1], 1)
end

# Condat algorithm
# https://lcondat.github.io/publis/Condat-fast_TV-SPL-2013.pdf
function tvnorm_prox_condat(y, x, lambda)
    # solves y = arg min_z lambda*sum_k |z_{k+1}-z_k| + 1/2 * ||z-x||^2
    N = length(x)

    k = k0 = kmin = kplus = 1
    vmin = x[1] - lambda
    vmax = x[1] + lambda
    umin = lambda
    umax = -lambda

    while 0 < 1
        while k == N
            if umin < 0
                y[k0:kmin] .= vmin
                kmin += 1
                k = k0 = kmin
                vmin = x[k]
                umin = lambda
                umax = x[k] + lambda - vmax
            elseif umax > 0
                y[k0:kplus] .= vmax
                kplus +=1
                k = k0 = kplus
                vmax = x[k]
                umax = -lambda
                umin = x[k] - lambda - vmin
            else
                y[k0:N] .= vmin + umin/(k-k0+1)
                return
            end

            if k==N
                y[N] = vmin + umin
                return
            end
        end


        if x[k+1] + umin < vmin - lambda
            y[k0:kmin] .= vmin
            kmin += 1
            k = k0 = kplus = kmin
            vmin = x[k]
            vmax = x[k] + 2*lambda
            umin = lambda
            umax = -lambda
        elseif x[k+1] + umax > vmax + lambda
            y[k0:kplus] .= vmax
            kplus += 1
            k = k0 =kmin = kplus
            vmin = x[k] - 2*lambda
            vmax = x[k]
            umin = lambda
            umax = -lambda
        else
            k += 1
            umin = umin + x[k] - vmin
            umax = umax + x[k] - vmax
            if umin >= lambda
                vmin = vmin + (umin-lambda)/(k-k0+1)
                umin = lambda
                kmin = k
            end

            if umax <= -lambda
                vmax += (umax+lambda)/(k-k0+1)
                umax = -lambda
                kplus = k
            end
        end
    end
end

function prox!(y, f::TotalVariation1D, x, gamma)
    a = gamma * f.lambda
    tvnorm_prox_condat(y, x, a)
    return f.lambda * norm(y[2:end] - y[1:end-1], 1)
end
