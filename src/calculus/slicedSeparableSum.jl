# Separable sum, using slices of an array as variables

export SlicedSeparableSum

"""
    SlicedSeparableSum((f_1, ..., f_k), (J_1, ..., J_k))

Return the function
```math
g(x) = \\sum_{i=1}^k f_i(x_{J_i}).
```

    SlicedSeparableSum(f, (J_1, ..., J_k))

Analogous to the previous one, but apply the same function `f` to all slices
of the variable `x`:
```math
g(x) = \\sum_{i=1}^k f(x_{J_i}).
```
"""
struct SlicedSeparableSum{S <: Tuple, T <: AbstractArray, N}
  fs::S    # Tuple, where each element is a Vector with elements of the same type; the functions to prox on
  # Example: S = Tuple{Array{ProximalOperators.NormL1{Float64},1}, Array{ProximalOperators.NormL2{Float64},1}}
  idxs::T  # Vector, where each element is a Vector containing the indices to prox on
  # Example: T = Array{Array{Tuple{Colon,UnitRange{Int64}},1},1}
end

function SlicedSeparableSum(fs::Tuple, idxs::Tuple)
    ftypes = DataType[]
    fsarr = Array{Any,1}[]
    indarr = Array{eltype(idxs),1}[]
    for (i,f) in enumerate(fs)
        t = typeof(f)
        fi = findfirst(isequal(t), ftypes)
        if fi === nothing
            push!(ftypes, t)
            push!(fsarr, Any[f])
            push!(indarr, eltype(idxs)[idxs[i]])
        else
            push!(fsarr[fi], f)
            push!(indarr[fi], idxs[i])
        end
    end
    fsnew = ((Array{typeof(fs[1]),1}(fs) for fs in fsarr)...,)
    @assert typeof(fsnew) == Tuple{(Array{ft,1} for ft in ftypes)...}
    SlicedSeparableSum{typeof(fsnew),typeof(indarr),length(fsnew)}(fsnew, indarr)
end

# Constructor for the case where the same function is applied to all slices
SlicedSeparableSum(f::F, idxs::T) where {F, T <: Tuple} =
SlicedSeparableSum(Tuple(f for k in eachindex(idxs)), idxs)

# Unroll the loop over the different types of functions to evaluate
@generated function (f::SlicedSeparableSum{A, B, N})(x) where {A, B, N}
    ex = :(v = 0.0)
    for i = 1:N # For each function type
        ex = quote $ex;
            for k in eachindex(f.fs[$i]) # For each function of that type
                v += f.fs[$i][k](view(x,f.idxs[$i][k]...))
            end
        end
    end
    ex = :($ex; return v)
end

# Unroll the loop over the different types of functions to prox on
@generated function prox!(y, f::SlicedSeparableSum{A, B, N}, x, gamma) where {A, B, N}
    ex = :(v = 0.0)
    for i = 1:N # For each function type
        ex = quote $ex;
            for k in eachindex(f.fs[$i]) # For each function of that type
                g = prox!(view(y, f.idxs[$i][k]...), f.fs[$i][k], view(x,f.idxs[$i][k]...), gamma)
                v += g
            end
        end
    end
    ex = :($ex; return v)
end

component_types(::Type{SlicedSeparableSum{S, T, N}}) where {S, T, N} = Tuple(A.parameters[1] for A in fieldtypes(S))

@generated is_prox_accurate(::Type{T}) where T <: SlicedSeparableSum = return all(is_prox_accurate, component_types(T)) ? :(true) : :(false)
@generated is_convex(::Type{T}) where T <: SlicedSeparableSum = return all(is_convex, component_types(T)) ? :(true) : :(false)
@generated is_set(::Type{T}) where T <: SlicedSeparableSum = return all(is_set, component_types(T)) ? :(true) : :(false)
@generated is_singleton(::Type{T}) where T <: SlicedSeparableSum = return all(is_singleton, component_types(T)) ? :(true) : :(false)
@generated is_cone(::Type{T}) where T <: SlicedSeparableSum = return all(is_cone, component_types(T)) ? :(true) : :(false)
@generated is_affine(::Type{T}) where T <: SlicedSeparableSum = return all(is_affine, component_types(T)) ? :(true) : :(false)
@generated is_smooth(::Type{T}) where T <: SlicedSeparableSum = return all(is_smooth, component_types(T)) ? :(true) : :(false)
@generated is_generalized_quadratic(::Type{T}) where T <: SlicedSeparableSum = return all(is_generalized_quadratic, component_types(T)) ? :(true) : :(false)
@generated is_strongly_convex(::Type{T}) where T <: SlicedSeparableSum = return all(is_strongly_convex, component_types(T)) ? :(true) : :(false)

function prox_naive(f::SlicedSeparableSum, x, gamma)
    fy = 0
    y = similar(x)
    for t in eachindex(f.fs)
        for k in eachindex(f.fs[t])
            y[f.idxs[t][k]...], fy1 = prox_naive(f.fs[t][k], x[f.idxs[t][k]...], gamma)
            fy += fy1
        end
    end
    return y, fy
end
