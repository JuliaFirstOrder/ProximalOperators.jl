export PointwiseMinimum

struct PointwiseMinimum{T <: Tuple} <: ProximableFunction
    fs::T
end

PointwiseMinimum(fs...) = PointwiseMinimum{typeof(fs)}(fs)

is_cone(g::PointwiseMinimum{T}) where T = all(is_cone(f) for f in g.fs)
is_set(g::PointwiseMinimum{T}) where T = all(is_set(f) for f in g.fs)

function (g::PointwiseMinimum{T})(x) where T
    return minimum(f(x) for f in g.fs)
end

function prox!(y::T, g::PointwiseMinimum, x::T, gamma::R=R(1)) where
    {R <: Real, C <: Union{R, Complex{R}}, T <: AbstractArray{C}}
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

function prox_naive(g::PointwiseMinimum, x::T, gamma::R=R(1)) where
    {R <: Real, C <: Union{R, Complex{R}}, T <: AbstractArray{C}}
    proxes = [prox_naive(f, x, gamma) for f in g.fs]
    moreau_envs = [f_y + R(1)/(R(2)*gamma)*norm(x - y)^2 for (y, f_y) in proxes]
    _, i_min = findmin(moreau_envs)
    y = proxes[i_min][1]
    return y, minimum(f(y) for f in g.fs)
end
