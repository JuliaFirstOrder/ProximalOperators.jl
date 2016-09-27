# Prox.jl [![Build Status](https://travis-ci.org/kul-forbes/Prox.jl.svg?branch=master)](https://travis-ci.org/kul-forbes/Prox.jl) [![Coverage Status](https://coveralls.io/repos/github/kul-forbes/Prox.jl/badge.svg?branch=master)](https://coveralls.io/github/kul-forbes/Prox.jl?branch=master)

Proximal operators for nonsmooth optimization in Julia.
This package can be used to easily implement proximal algorithms
for convex and nonconvex optimization problems such as ADMM,
the alternating direction method of multipliers.

## Installation

From the Julia command line `Pkg.clone("https://github.com/kul-forbes/Prox.jl.git")`.
Use `Pkg.update()` to keep the package up to date.

## Usage

With `using Prox` the package exports the `prox` and `prox!` methods to evaluate
the proximal mapping of several functions. Such functions can be instantiated using
*constructors*. For example, you can create the L1-norm as follows:

```julia
julia> f = NormL1(3.5)
description : L1 norm
type        : Array{Complex} → Real
expression  : x ↦ λ||x||_1
parameters  : λ = 3.5
```

For the available constructors, see the [dedicated section](https://github.com/kul-forbes/Prox.jl#available-functions).
Functions created this way are, of course, callable:

```julia
julia> x = randn(10) # some random point
julia> f(x)
32.40700818735099
```

### `prox` and `prox!`

The `prox` method evaluates the proximal operator associated with a function, given a
point and (optionally) a positive stepsize parameter.
This returns the proximal point `y` and the value of the function at `y`:

```julia
julia> y, fy = prox(f, x, 0.5)
```

The `prox!` method evaluates the proximal operator *in place*, and only returns the
function value at the proximal point.

```julia
julia> fx = prox!(f, x, 0.5) # in-place equivalent to x, fx = prox(f, x, 0.5)
```

## Functions

The available constructors are listed in the following table.
You can access the specific documentation of each of them from the command line
of Julia (try typing in `?NormL1`) to have information on their parameters.

Function        | Description                                          | Properties
----------------|------------------------------------------------------|----------------
`IndAffine`     | Indicator of an affine subspace                      | convex
`IndBallInf`    | Indicator of an infinity-norm ball                   | convex
`IndBallL0`     | Indicator of an L0 pseudo-norm ball                  | nonconvex
`IndBallL2`     | Indicator of a Euclidean ball                        | convex
`IndBallRank`   | Indicator of the set of matrices with given rank     | nonconvex
`IndBox`        | Indicator of a box                                   | convex
`IndHalfspace`  | Indicator of a halfspace                             | convex
`IndNonnegative`| Indicator of the nonnegative orthant                 | convex
`IndNonpositive`| Indicator of the nonpositive orthant                 | convex
`IndSimplex`    | Indicator of the probability simplex                 | convex
`IndSOC`        | Indicator of the second-order cone                   | convex
`IndSphereL2`   | Indicator of Euclidean sphere                        | nonconvex
`ElasticNet`    | Elastic-net regularization                           | convex
`NormL0`        | L0 pseudo-norm                                       | nonconvex
`NormL1`        | L1 norm                                              | convex
`NormL2`        | Euclidean norm                                       | convex
`NormL21`       | Sum-of-L2 norms                                      | convex
`NuclearNorm`   | Nuclear norm                                         | convex
`SqrNormL2`     | Squared Euclidean norm                               | convex
`DistL2`        | Euclidean distance from a convex set                 | convex
`SqrDistL2`     | Squared Euclidean distance from a convex set         | convex

## References

1. Neal Parikh and Stephen Boyd, [*Proximal Algorithms*](http://dx.doi.org/10.1561/2400000003).
Foundations and Trends in Optimization 1, 3 (2014), 127-239.

## Credits

Prox.jl is developed by [Lorenzo Stella](https://lostella.github.io) and [Niccolò Antonello](http://homes.esat.kuleuven.be/~nantonel/) at [KU Leuven, ESAT/Stadius](https://www.esat.kuleuven.be/stadius/).
