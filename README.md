# Prox.jl

[![Build Status](https://travis-ci.org/kul-forbes/Prox.jl.svg?branch=master)](https://travis-ci.org/kul-forbes/Prox.jl) [![Build status](https://ci.appveyor.com/api/projects/status/00rk6ip0y0t2wj8t?svg=true)](https://ci.appveyor.com/project/lostella/prox-jl) [![Coverage Status](https://coveralls.io/repos/github/kul-forbes/Prox.jl/badge.svg?branch=master)](https://coveralls.io/github/kul-forbes/Prox.jl?branch=master)


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

For the available constructors, see the [dedicated section](https://github.com/kul-forbes/Prox.jl#functions).
Functions created this way are, of course, callable:

```julia
julia> x = randn(10) # some random point
julia> f(x)
32.40700818735099
```

**`prox`** evaluates the proximal operator associated with a function, given a point and (optionally) a positive stepsize parameter,
returning the proximal point `y` and the value of the function at `y`:

```julia
julia> y, fy = prox(f, x, 0.5) # last argument is 1.0 if absent
```

**`prox!`** evaluates the proximal operator *in place*, and only returns the function value at the proximal point:

```julia
julia> fx = prox!(f, x, 0.5) # in-place equivalent to x, fx = prox(f, x, 0.5)
julia> fy = prox!(f, x, y, 0.5) # in-place equivalent to y, fy = prox(f, x, 0.5)
```

## Functions

The available constructors are listed in the following tables.
You can access the specific documentation of each of them from the command line
of Julia (try typing in `?NormL1`) to have information on their parameters.

### Indicator functions

Name            | Type of set                         | Properties
----------------|-------------------------------------|----------------
`IndAffine`     | Affine subspace                     | convex
`IndBallInf`    | Infinity-norm ball                  | convex
`IndBallL0`     | L0 pseudo-norm ball                 | nonconvex
`IndBallL1`     | L1 norm ball                        | convex
`IndBallL2`     | Euclidean ball                      | convex
`IndBallRank`   | Set of matrices with given rank     | nonconvex
`IndBox`        | Box                                 | convex
`IndFree`       | Indicator of the free cone          | convex
`IndHalfspace`  | Halfspace                           | convex
`IndNonnegative`| Nonnegative orthant                 | convex
`IndNonpositive`| Nonpositive orthant                 | convex
`IndPoint`      | Indicator of a singleton            | convex
`IndPSD`        | Positive semidefinite cone          | convex
`IndSimplex`    | Simplex                             | convex
`IndSOC`        | Second-order cone                   | convex
`IndSphereL2`   | Euclidean sphere                    | nonconvex
`IndZero`       | Indicator of the zero singleton     | convex

### Norms, pseudo-norms, regularization functions

Name            | Description                         | Properties
----------------|-------------------------------------|----------------
`ElasticNet`    | Elastic-net regularization          | convex
`NormL0`        | L0 pseudo-norm                      | nonconvex
`NormL1`        | L1 norm                             | convex
`NormL2`        | Euclidean norm                      | convex
`NormL21`       | Sum-of-L2 norms                     | convex
`NuclearNorm`   | Nuclear norm                        | convex
`SqrNormL2`     | Squared Euclidean norm              | convex

### Penalties

Name            | Description                         | Properties
----------------|-------------------------------------|-----------------
`HingeLoss`     | Hinge loss function                 | convex
`LogBarrier`    | Logarithmic barrier                 | convex

### Distances

Name            | Description                                          | Properties
----------------|------------------------------------------------------|----------------
`DistL2`        | Euclidean distance from a convex set                 | convex
`SqrDistL2`     | Squared Euclidean distance from a convex set         | convex

## References

1. Neal Parikh and Stephen Boyd, [*Proximal Algorithms*](http://dx.doi.org/10.1561/2400000003).
Foundations and Trends in Optimization 1, 3 (2014), 127-239.

## Credits

Prox.jl is developed by [Lorenzo Stella](https://lostella.github.io) and [Niccolò Antonello](http://homes.esat.kuleuven.be/~nantonel/) at [KU Leuven, ESAT/Stadius](https://www.esat.kuleuven.be/stadius/).
