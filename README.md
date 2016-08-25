# Prox.jl

Nonsmooth functions and proximal operators in Julia.

## Installation

From the Julia command line `Pkg.clone("https://github.com/kul-forbes/Prox.jl.git")`.
Remember to update the package with `Pkg.update()`.

## Usage

With `using Prox` the package exports constructors to create functions, and the
`prox` method to evaluate their proximal mapping. For example, you can create
the L1-norm as follows:

```julia
julia> f = NormL1(3.5)
description : L1 norm
type        : C^n → R
expression  : x ↦ λ||x||_1
parameters  : λ = 3.5
```

Functions created this way are, of course, callable:

```julia
julia> x = randn(10) # some random point
julia> f(x)
32.40700818735099
```

The `prox` method evaluates the proximal operator associated with a function, given a
positive stepsize parameter and a point. The return values are the proximal point
`y` and the value of the function at `y`:

```julia
julia> y, fy = prox(f, 0.5, x)
```

## Available functions

The available constructors are described in the following table.
You can access the specific documentation of each of them from the command line
of Julia (try typing in `?NormL1`) to have information on their parameters.

Function        | Description                                          | Properties
----------------|------------------------------------------------------|----------------
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
`ElasticNet`    | Elastic-net regularization                           | convex
`NormL0`        | L0 pseudo-norm                                       | nonconvex
`NormL1`        | L1 norm                                              | convex
`NormL2`        | Euclidean norm                                       | convex
`NormL21`       | Sum-of-L2 norms                                      | convex
`SqrNormL2`     | Squared Euclidean norm                               | convex
`DistL2`        | Euclidean distance from a convex set                 | convex
`SqrDistL2`     | Squared Euclidean distance from a convex set         | convex

## References

1. Neal Parikh and Stephen Boyd, *Proximal Algorithms*.
Foundations and Trends in Optimization 1, 3 (January 2014), 127-239. http://dx.doi.org/10.1561/2400000003

## Credits

Prox.jl is developed by [Lorenzo Stella](https://lostella.github.io) and [Niccolò Antonello](http://homes.esat.kuleuven.be/~nantonel/) at [KU Leuven, ESAT/Stadius](https://www.esat.kuleuven.be/stadius/).
