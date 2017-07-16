# Prox and gradient

The following methods allow to evaluate the proximal mapping (and gradient, when defined) of mathematical functions, which are constructed according to what described in [Functions](@ref) and [Calculus rules](@ref).

```@docs
prox
prox!
gradient
gradient!
```

## Complex and matrix variables

The proximal mapping is usually discussed in the case of functions over ``\mathbb{R}^n``. However, by adapting the inner product ``\langle\cdot,\cdot\rangle`` and associated norm ``\|\cdot\|`` adopted in its definition, one can extend the concept to functions over more general spaces.
When functions of unidimensional arrays (vectors) are concerned, the standard Euclidean product and norm are used in defining [`prox`](@ref) (therefore [`prox!`](@ref), but also [`gradient`](@ref) and [`gradient!`](@ref)).
This are the inner product and norm which are computed by `dot` and `norm` in Julia.

When bidimensional, tridimensional (matrices and tensors) and higher dimensional arrays are concerned, then the definitions of proximal mapping and gradient are naturally extended by considering the appropriate inner product.
For ``k``-dimensional arrays, of size ``n_1 \times n_2 \times \ldots \times n_k``, we consider the inner product
```math
\langle A, B \rangle = \sum_{i_1,\ldots,i_k} A_{i_1,\ldots,i_k} \cdot B_{i_1,\ldots,i_k}
```
which reduces to the usual Euclidean product in case of unidimensional arrays, and to the *trace product* ``\langle A, B \rangle = \mathrm{tr}(A^\top B)`` in the case of matrices (bidimensional arrays). This inner product, and the associated norm, are the ones computed by `vecdot` and `vecnorm` in Julia.

## Multiple variable blocks

By combining functions together through [`SeparableSum`](@ref), the resulting function will have multiple inputs, *i.e.*, it will be defined over the *Cartesian product* of the domains of the individual functions.
To represent elements (points) of such product space, here we use Julia's `Tuple` objects.

**Example.** Suppose that the following function needs to be represented:
```math
f(x, Y) = \|x\|_1 + \|Y\|_*,
```
that is, the sum of the ``L_1`` norm of some vector ``x`` and the nuclear norm (the sum of the singular values) of some matrix ``Y``. This is accomplished as follows:
```example blocks
using ProximalOperators
f = SeparableSum(NormL1(), NuclearNorm());
```
Now, function `f` is defined over *pairs* of appropriate `Array` objects. Likewise, the [`prox`](@ref) method will take pairs of `Array`s as inputs, and return pairs of `Array`s as output:
```example block
x = randn(10); # some random vector
Y = randn(20, 30); # some random matrix
f_xY = f((x, Y)); # evaluates f at (x, Y)
(u, V), f_uV = prox(f, (x, Y), 1.3); # computes prox at (x, Y)
```
The same holds for the separable sum of more than two functions, in which case "pairs" are to be replaced with `Tuple`s of the appropriate length.
