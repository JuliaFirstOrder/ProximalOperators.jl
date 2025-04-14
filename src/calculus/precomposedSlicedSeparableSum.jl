# Separable sum, using slices of an array as variables

export PrecomposedSlicedSeparableSum

"""
    precomposedSlicedSeparableSum((f_1, ..., f_k), (J_1, ..., J_k), (L_1, ..., L_k))

Return the function
```math
g(x) = \\sum_{i=1}^k f_i(L_i * x_{J_i}).
```

    precomposedSlicedSeparableSum(f, (J_1, ..., J_k), (L_1, ..., L_k))

Analogous to the previous one, but apply the same function `f` to all slices
of the variable `x`:
```math
g(x) = \\sum_{i=1}^k f(L_i * x_{J_i}).
```
"""
struct PrecomposedSlicedSeparableSum{S <: Tuple, T <: AbstractArray, U <: AbstractArray, V <: AbstractArray, N}
  fs::S    # Tuple, where each element is a Vector with elements of the same type; the functions to prox on
  # Example: S = Tuple{Array{ProximalOperators.NormL1{Float64},1}, Array{ProximalOperators.NormL2{Float64},1}}
  idxs::T  # Vector, where each element is a Vector containing the indices to prox on
  # Example: T = Array{Array{Tuple{Colon,UnitRange{Int64}},1},1}
  ops::U   # Vector of operations (matrices or AbstractOperators) to apply to the function
  # Example: U = Array{Array{Matrix{Float64},1},1}
  μs::V   # Vector of mu values for each function
end

function PrecomposedSlicedSeparableSum(fs::Tuple, idxs::Tuple, ops::Tuple, μs::Tuple) 
    @assert length(fs) == length(idxs)
    @assert length(fs) == length(ops)
    ftypes = DataType[]
    fsarr = Array{Any,1}[]
    indarr = Array{eltype(idxs),1}[]
    opsarr = Array{Any,1}[]
    μsarr = Array{Any,1}[]
    for (i,f) in enumerate(fs)
        t = typeof(f)
        fi = findfirst(isequal(t), ftypes)
        if fi === nothing
            push!(ftypes, t)
            push!(fsarr, Any[f])
            push!(indarr, eltype(idxs)[idxs[i]])
            push!(opsarr, Any[ops[i]])
            push!(μsarr, Any[μs[i]])
        else
            push!(fsarr[fi], f)
            push!(indarr[fi], idxs[i])
            push!(opsarr[fi], ops[i])
            push!(μsarr[fi], μs[i])
        end
    end
    fsnew = ((Array{typeof(fs[1]),1}(fs) for fs in fsarr)...,)
    @assert typeof(fsnew) == Tuple{(Array{ft,1} for ft in ftypes)...}
    PrecomposedSlicedSeparableSum{typeof(fsnew),typeof(indarr),typeof(opsarr),typeof(μsarr),length(fsnew)}(fsnew, indarr, opsarr, μsarr)
end

# Constructor for the case where the same function is applied to all slices
PrecomposedSlicedSeparableSum(f::F, idxs::T, ops::U, μs::V) where {F, T <: Tuple, U <: Tuple, V <: Tuple} =
    PrecomposedSlicedSeparableSum(Tuple(f for k in eachindex(idxs)), idxs, ops, μs)

# Unroll the loop over the different types of functions to evaluate
function (f::PrecomposedSlicedSeparableSum)(x::Tuple)
    v = zero(eltype(x[1]))
    for (fs_group, idxs_group, ops_group) = zip(f.fs, f.idxs, f.ops) # For each function type
        for (fun, idx_group, hcat_op) in zip(fs_group, idxs_group, ops_group) # For each function of that type
            for (var_index, (x_var, idx)) in enumerate(zip(x, idx_group))
                if idx isa Tuple
                    v += fun(hcat_op[var_index] * view(x_var, idx...))
                elseif idx isa Colon
                    v += fun(hcat_op[var_index] * x_var)
                elseif idx isa Nothing
                    # do nothing
                else
                    v += fun(hcat_op[var_index] * view(x_var, idx))
                end
            end
        end
    end
    return v
end

function slice_var(x, idx)
    if idx isa Tuple
        return view(x, idx...)
    elseif idx isa Colon
        return x
    elseif idx isa Nothing
        return similar(x)
    else
        return view(x, idx)
    end
end

# Unroll the loop over the different types of functions to prox on
function prox!(y::Tuple, f::PrecomposedSlicedSeparableSum, x::Tuple, gamma)
    v = zero(eltype(x[1]))
    counter = 1
    for (fs_group, idxs_group, ops_group, μ_group) = zip(f.fs, f.idxs, f.ops, f.μs) # For each function type
        for (fun, idx_group, hcat_op, μ) in zip(fs_group, idxs_group, ops_group, μ_group) # For each function of that type
            sliced_x = Tuple(slice_var(x_var, idx) for (x_var, idx) in zip(x, idx_group))
            sliced_y = Tuple(slice_var(y_var, idx) for (y_var, idx) in zip(y, idx_group))
            res = hcat_op * sliced_x
            prox_res, g = prox(fun, res, μ.*gamma)
            prox_res .-= res
            prox_res ./= μ
            mul!(sliced_y, adjoint(hcat_op), prox_res)
            for i in eachindex(sliced_x)
                sliced_y[i] .+= sliced_x[i]
            end
            v += g
            counter += 1
        end
    end
    return v
end

component_types(::Type{PrecomposedSlicedSeparableSum{S, T, N}}) where {S, T, N} = Tuple(A.parameters[1] for A in fieldtypes(S))

@generated is_proximable(::Type{T}) where T <: PrecomposedSlicedSeparableSum = return all(is_proximable, component_types(T)) ? :(true) : :(false)
@generated is_convex(::Type{T}) where T <: PrecomposedSlicedSeparableSum = return all(is_convex, component_types(T)) ? :(true) : :(false)
@generated is_set_indicator(::Type{T}) where T <: PrecomposedSlicedSeparableSum = return all(is_set_indicator, component_types(T)) ? :(true) : :(false)
@generated is_singleton_indicator(::Type{T}) where T <: PrecomposedSlicedSeparableSum = return all(is_singleton_indicator, component_types(T)) ? :(true) : :(false)
@generated is_cone_indicator(::Type{T}) where T <: PrecomposedSlicedSeparableSum = return all(is_cone_indicator, component_types(T)) ? :(true) : :(false)
@generated is_affine_indicator(::Type{T}) where T <: PrecomposedSlicedSeparableSum = return all(is_affine_indicator, component_types(T)) ? :(true) : :(false)
@generated is_smooth(::Type{T}) where T <: PrecomposedSlicedSeparableSum = return all(is_smooth, component_types(T)) ? :(true) : :(false)
@generated is_generalized_quadratic(::Type{T}) where T <: PrecomposedSlicedSeparableSum = return all(is_generalized_quadratic, component_types(T)) ? :(true) : :(false)
@generated is_strongly_convex(::Type{T}) where T <: PrecomposedSlicedSeparableSum = return all(is_strongly_convex, component_types(T)) ? :(true) : :(false)

function prox_naive(f::PrecomposedSlicedSeparableSum, x, gamma)
    fy = 0
    y = similar.(x)
    for (fs_group, idxs_group, ops_group, μ_group) = zip(f.fs, f.idxs, f.ops, f.μs) # For each function type
        for (fun, idx_group, hcat_op, μ) in zip(fs_group, idxs_group, ops_group, μ_group) # For each function of that type
            sliced_x = Tuple(slice_var(x_var, idx) for (x_var, idx) in zip(x, idx_group))
            sliced_y = Tuple(slice_var(y_var, idx) for (y_var, idx) in zip(y, idx_group))
            res = hcat_op * sliced_x
            prox_res, _fy = prox_naive(fun, res, μ.*gamma)
            prox_res = (prox_res .- res) ./ μ
            mul!(sliced_y, adjoint(hcat_op), prox_res)
            fy += _fy
            for i in eachindex(sliced_x)
                sliced_y[i] .+= sliced_x[i]
            end
        end
    end
    return y, fy
end
