# Separable sum, using slices of an array as variables

export SlicedSeparableSum

"""
**Sliced separable sum of functions**

    SlicedSeparableSum((f₁,…,fₖ), (J₁,…,Jₖ))

Returns the function
```math
g(x) = ∑_\{i=1\}^k f_i(x_{J_i}).
```

    SlicedSeparableSum(f, (J₁,…,Jₖ))

Analogous to the previous one, but applies the same function `f` to all slices of the variable `x`:
```math
g(x) = ∑_\{i=1\}^k f(x_{J_i}).
```
"""

struct SlicedSeparableSum{S <: Tuple, T <: AbstractArray, N} <: ProximableFunction
  fs::S    # Tuple, where each element is a Vector with elements of the same type; the functions to prox on
  # Example: S = Tuple{Array{ProximalOperators.NormL1{Float64},1}, Array{ProximalOperators.NormL2{Float64},1}}
  idxs::T  # Vector, where each element is a Vector contining the indices to prox on
  # Example: T = Array{Array{Tuple{Colon,UnitRange{Int64}},1},1}
end

function SlicedSeparableSum(fs::S, idxs::T) where {N,
						   S <: Tuple{Vararg{<:ProximableFunction,N}},
						   M, 
						   I <: Integer, 
						   T1 <: NTuple{M,Union{I,
								       AbstractArray{I},
								       Colon,
								       Range
								       }
							       },
						   T <:NTuple{N,T1}
						   }
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

# Constructor for the case where the same function is applied to all slices
SlicedSeparableSum(f::F, idxs::T) where {F <: ProximableFunction, T <: Tuple} = 
SlicedSeparableSum(([f for k in eachindex(idxs)]...), idxs)

# Unroll the loop over the different types of functions to evaluate
@generated function (f::SlicedSeparableSum{A, B, N})(x::T) where {A, B, N, T <: AbstractArray}
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
@generated function prox!(y::AbstractArray{T}, f::SlicedSeparableSum{A, B, N}, x::AbstractArray{T}, gamma::Real=1.0) where {T <: RealOrComplex, A, B, N}
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
