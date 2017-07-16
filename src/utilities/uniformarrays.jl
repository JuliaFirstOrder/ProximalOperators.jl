# utilities to handle scalars as uniform arrays
# (instead writing multiple implementations of functions, proxes, gradients)

get_kth_elem{N <: Number}(n::N, k) = n
get_kth_elem{T}(n::T, k) = n[k]
