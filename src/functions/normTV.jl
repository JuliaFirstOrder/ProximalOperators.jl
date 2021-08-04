# Total variation norm (times a constant)

export NormTV

"""
** 1-dimensional ``TV`` norm**

    NormTV(λ=1)

With a nonnegative scalar parameter λ, returns the function
```math
f(x) = λ ∑_{i=2}^{n} |x_i - x_{i-1}|.
```
"""
struct NormTV{T <: Real} <: ProximableFunction
    lambda::T
    function NormTV{T}(lambda::T) where {T <: Real}
        if lambda < 0
            error("parameter λ must be nonnegative")
        else
            new(lambda)
        end
    end
end

is_separable(f::NormTV) = false
is_convex(f::NormTV) = true
is_positively_homogeneous(f::NormTV) = true

NormTV(lambda::R=1) where {R <: Real} = NormTV{R}(lambda)

function (f::NormTV)(x::AbstractArray)
    return f.lambda * norm(x[2:end] - x[1:end-1], 1)
end

# Condat algorithm
# https://lcondat.github.io/publis/Condat-fast_TV-SPL-2013.pdf
function tvnorm_prox_condat(y::AbstractArray, x::AbstractArray, lambda::Real)
    # solves y = arg min_z lambda*sum_k |z_{k+1}-z_k| + 1/2 * ||z-x||^2
    N = length(x);

    k=k0=kmin=kplus=1;
    vmin = x[1] - lambda;
    vmax = x[1] + lambda;
    umin = lambda;
    umax = -lambda;

    while 0 < 1
        while k == N
            if umin < 0
                y[k0:kmin] .= vmin;
                kmin += 1;
                k = k0 = kmin;
                vmin = x[k]; umin = lambda;
                umax = x[k] + lambda - vmax;
            elseif umax > 0
                y[k0:kplus] .= vmax;
                kplus +=1;
                k=k0=kplus;
                vmax = x[k]; umax = -lambda;
                umin = x[k] - lambda - vmin;
            else
                y[k0:N] .= vmin + umin/(k-k0+1);
                return
            end

            if k==N
                y[N] = vmin + umin;
                return
            end
        end


        if x[k+1] + umin < vmin - lambda
            y[k0:kmin] .= vmin;
            kmin +=1;
            k = k0 =kplus =kmin;
            vmin = x[k]; vmax = x[k] + 2*lambda;
            umin = lambda; umax = -lambda;
        elseif x[k+1] + umax > vmax + lambda
            y[k0:kplus] .= vmax;
            kplus +=1;
            k = k0 =kmin = kplus;
            vmin = x[k] - 2*lambda; vmax = x[k];
            umin = lambda; umax = -lambda;
        else
            k +=1 ;
            umin = umin + x[k] - vmin;
            umax = umax + x[k] - vmax;
            if umin >= lambda
                vmin = vmin + (umin-lambda)/(k-k0+1);
                umin = lambda; kmin = k;
            end

            if umax <= -lambda
                vmax += (umax+lambda)/(k-k0+1);
                umax = -lambda; kplus = k;
            end
        end
    end
end

function prox!(y::AbstractArray{T}, f::NormTV, x::AbstractArray{T}, gamma::Real=1.0) where T <: Real
    a = gamma * f.lambda
    tvnorm_prox_condat(y, x, a)
    return f.lambda * norm(y[2:end] - y[1:end-1], 1)
end

fun_name(f::NormTV) = "1D Total variation norm"
fun_dom(f::NormTV) = "AbstractArray{Real}"
fun_expr(f::NormTV) = "x ↦ λ ∑_{i=2}^{n} |x_i - x_{i-1}|"
fun_params(f::NormTV) = "λ = $(f.lambda)"
