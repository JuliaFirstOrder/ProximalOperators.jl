# Separable sum, using slices of an array as variables

immutable SlicedSeparableSum{S <: Tuple, T <: AbstractArray, N} <: ProximableFunction
  fs::S    # Tuple, where each element is a Vector with elements of the same type; the functions to prox on
  # Example: S = Tuple{Array{ProximalOperators.NormL1{Float64},1}, Array{ProximalOperators.NormL2{Float64},1}}
  idxs::T  # Vector, where each element is a Vector contining the indices to prox on
  # Example: T = Array{Array{Tuple{Colon,UnitRange{Int64}},1},1}
end

function SlicedSeparableSum{S <: AbstractArray, T <: AbstractArray}(fs::S, idxs::T)
  if size(fs) != size(idxs)
    error("size(fs) must coincide with size(idxs)")
  else
    for k in eachindex(idxs)
      cond = [typeof(t) <: Integer || typeof(t) <: AbstractArray{Integer,1} || typeof(t) <: Colon || typeof(t) <: Range for t in idxs[k]]
			!all(cond) ? error("invalid index $(k)") : nothing
    end
    ftypes = DataType[]
    fsarr = Array{Any,1}[]
    indarr = Array{eltype(idxs),1}[]
    for (i,f) in enumerate(fs)
      t = typeof(f)
      fi = findfirst(ftypes, t)
      if fi == 0
        push!(ftypes, t)
        push!(fsarr, Any[f])
        push!(indarr, eltype(idxs)[idxs[i]])
      else
        push!(fsarr[fi], f)
        push!(indarr[fi], idxs[i])
      end
    end
    fsnew = ((Array{typeof(fs[1]),1}(fs) for fs in fsarr)...)
    @assert typeof(fsnew) == Tuple{(Array{ft,1} for ft in ftypes)...}
    SlicedSeparableSum{typeof(fsnew),typeof(indarr),length(fsnew)}(fsnew, indarr)
  end
end

# Constructor for the case where the same function is applied to all slices
SlicedSeparableSum{F <: ProximableFunction, T <: AbstractArray}(f::F, idxs::T) = SlicedSeparableSum([f for k in eachindex(idxs)], idxs)

# Unroll the loop over the different types of functions to evaluate
@generated function (f::SlicedSeparableSum{A, B, N}){A, B, N, T <: AbstractArray}(x::T)
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
@generated function prox!{T <: RealOrComplex, A, B, N}(y::AbstractArray{T}, f::SlicedSeparableSum{A, B, N}, x::AbstractArray{T}, gamma::Real=1.0)
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

is_prox_accurate(f::SlicedSeparableSum) = all([all(is_prox_accurate.(f.fs[k])) for k in eachindex(f.fs)])
is_convex(f::SlicedSeparableSum) = all([all(is_convex.(f.fs[k])) for k in eachindex(f.fs)])
is_set(f::SlicedSeparableSum) = all([all(is_set.(f.fs[k])) for k in eachindex(f.fs)])
is_cone(f::SlicedSeparableSum) = all([all(is_cone.(f.fs[k])) for k in eachindex(f.fs)])

fun_name(f::SlicedSeparableSum) = "sliced separable sum"
fun_dom(f::SlicedSeparableSum) = "n/a" # for now
fun_expr(f::SlicedSeparableSum) = "hard to explain"
fun_params(f::SlicedSeparableSum) = "n/a" # for now

function prox_naive(f::SlicedSeparableSum, x::AbstractArray, gamma)
	fy = 0;
	y = similar(x);
	for t in eachindex(f.fs)
		for k in eachindex(f.fs[t])
			y[f.idxs[t][k]...], fy1 = prox_naive(f.fs[t][k], x[f.idxs[t][k]...], gamma)
			fy += fy1
		end
	end
	return y, fy
end
