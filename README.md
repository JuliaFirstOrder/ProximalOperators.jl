# Prox.jl

Nonsmooth functions and proximal operators in Julia.

## Installation

From the Julia command line `Pkg.clone("https://github.com/lostella/Prox.jl.git")`.
Remember to update the package with `Pkg.update()`.

## Usage

Load Prox.jl with `using Prox`. The package exports constructors you can use to
instantiate functions, and the `prox` method to evaluate their proximal mapping.
The available constructors are described in the following table.

Function        | Description                                          | Properties
----------------|------------------------------------------------------|----------------
`ElasticNet`    | Elastic-net regularization                           | convex
`IndAffine`     | Indicator of an affine subspace                      | convex
`IndBallInf`    | Indicator of an infinity-norm ball                   | convex
`IndBallL0`     | Indicator of an L0 pseudo-norm ball                  | nonconvex
`IndBallL2`     | Indicator of an Euclidean ball                       | convex
`IndBallRank`   | Indicator of the set of matrices with given rank     | nonconvex
`IndBox`        | Indicator of a box                                   | convex
`IndHalfspace`  | Indicator of a halfspace                             | convex
`IndNonnegative`| Indicator of the nonnegative orthant                 | convex
`IndSimplex`    | Indicator of the probability simplex                 | convex
`IndSOC`        | Indicator of the second-order cone                   | convex
`NormL0`        | L0 pseudo-norm                                       | nonconvex
`NormL1`        | L1 norm                                              | convex
`NormL2`        | Euclidean norm                                       | convex
`NormL21`       | Sum-of-L2 norms                                      | convex
`SqrNormL2`     | Squared Euclidean norm                               | convex

Each function can be customized with parameters: you can access the specific documentation
of each function from the command line of Julia directly (try typing in `?NormL1`).

Once a function has been created, you can at any time inspect it by simply printing it out:

```
julia> f = NormL1(3.5)
description : weighted L1 norm
type        : C^n → R
expression  : x ↦ λ||x||_1
parameters  : λ = 3.5
```

Functions created this way are, of course, callable:

```
julia> x = randn(10)
julia> f(x)
32.40700818735099
```

The `prox` method evaluates the proximal operator associated with a function, given a
positive stepsize parameter and a point. The return values are the proximal point
`y` and the value of the function at `y`:

```
julia> y, fy = prox(f, 0.5, x)
```

## References

PUT SOME REFERENCES HERE

## Credits

Prox.jl is developed by [Lorenzo Stella](https://lostella.github.io) and [Niccolò Antonello](http://homes.esat.kuleuven.be/~nantonel/) at [KU Leuven, ESAT/Stadius](https://www.esat.kuleuven.be/stadius/).
