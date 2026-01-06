export PointwiseMinimum

"""
    PointwiseMinimum(f_1, ..., f_k)

Given functions `f_1` to `f_k`, return their pointwise minimum, that is function
```math
g(x) = \\min\\{f_1(x), ..., f_k(x)\\}
```
Note that `g` is a nonconvex function in general.
"""
struct PointwiseMinimum{T}
    fs::T
end

PointwiseMinimum(fs...) = PointwiseMinimum{typeof(fs)}(fs)

component_types(::Type{PointwiseMinimum{T}}) where T = fieldtypes(T)

@generated is_set_indicator(::Type{T}) where T <: PointwiseMinimum = return all(is_set_indicator, component_types(T)) ? :(true) : :(false)
@generated is_cone_indicator(::Type{T}) where T <: PointwiseMinimum = return all(is_cone_indicator, component_types(T)) ? :(true) : :(false)

function (g::PointwiseMinimum{T})(x) where T
    return minimum(f(x) for f in g.fs)
end

function prox!(y, g::PointwiseMinimum, x, gamma)
    R = real(eltype(x))
    y_temp = similar(y)
    minimum_moreau_env = Inf
    for f in g.fs
        f_y_temp = prox!(y_temp, f, x, gamma)
        moreau_env = f_y_temp + R(1)/(2*gamma)*norm(x - y_temp)^2
        if moreau_env <= minimum_moreau_env
            copyto!(y, y_temp)
            minimum_moreau_env = moreau_env
        end
    end
    return g(y)
end

function prox_naive(g::PointwiseMinimum, x, gamma)
    R = real(eltype(x))
    proxes = [prox_naive(f, x, gamma) for f in g.fs]
    moreau_envs = [f_y + R(1)/(R(2)*gamma)*norm(x - y)^2 for (y, f_y) in proxes]
    _, i_min = findmin(moreau_envs)
    y = proxes[i_min][1]
    return y, minimum(f(y) for f in g.fs)
end
