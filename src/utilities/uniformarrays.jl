# utilities to handle scalars as uniform arrays
# (instead writing multiple implementations of functions, proxes, gradients)

get_kth_elem(n::N, k) where {N <: Number} = n
get_kth_elem(n::T, k) where {T} = n[k]
