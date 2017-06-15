# Prox and gradient

```@docs
prox
prox!
gradient
gradient!
```

## Complex and matrix variables

The proximal mapping is usually discussed in the case of functions over ``\mathbb{R}^n``. However, by adapting the inner product ``\langle\cdot,\cdot\rangle`` and associated norm ``\|\cdot\|`` adopted in its definition, one can extend the concept to functions over more general spaces.
When functions of unidimensional arrays (vectors) are concerned, the standard Euclidean product and norm are used in defining ``\mathrm{prox}_{\gamma f}`` (and therefore [`prox`](@ref), [`prox!`](@ref) but also [`gradient`](@ref), [`gradient!`](@ref)).
This are the inner product and norm which are computed by `dot` and `norm` in Julia.

When bidimensional, tridimensional (matrices and tensors) and higher dimensional arrays are concerned, then the definitions of proximal mapping and gradient are naturally extended by considering the appropriate inner product.
For ``k``-dimensional arrays, of size ``n_1 \times n_2 \times \ldots \times n_k``, we consider the inner product
```math
\langle A, B \rangle = \sum_{i_1,\ldots,i_k} A_{i_1,\ldots,i_k} \cdot B_{i_1,\ldots,i_k}
```
which reduces to the usual Euclidean product in case of unidimensional arrays, and to the *trace product* ``\langle A, B \rangle = \mathrm{tr}(A^\top B)`` in the case of matrices (bidimensional arrays). This inner product, and the associated norm, are the ones computed by `vecdot` and `vecnorm` in Julia.

## Blocks of variables
