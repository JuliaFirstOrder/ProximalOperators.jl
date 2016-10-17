## Functions

The available constructors are listed in the following tables.
You can access the specific documentation of each of them from the command line
of Julia (try typing in `?NormL1`) to have information on their parameters.

### Indicator functions

Name            | Type of set                             | Properties
----------------|-----------------------------------------|----------------
`IndAffine`     | Affine subspace                         | convex
`IndBallLinf`   | L-infinity norm ball                    | convex
`IndBallL0`     | L0 pseudo-norm ball                     | nonconvex
`IndBallL1`     | L1 norm ball                            | convex
`IndBallL2`     | Euclidean ball                          | convex
`IndBallRank`   | Set of matrices with given rank         | nonconvex
`IndBox`        | Box                                     | convex
`IndExpPrimal`  | Indicator of (primal) exponential cone  | convex
`IndExpDual`    | Indicator of (dual) exponential cone    | convex
`IndFree`       | Indicator of the free cone              | convex
`IndHalfspace`  | Halfspace                               | convex
`IndNonnegative`| Nonnegative orthant                     | convex
`IndNonpositive`| Nonpositive orthant                     | convex
`IndPoint`      | Indicator of a singleton                | convex
`IndPSD`        | Positive semidefinite cone              | convex
`IndSimplex`    | Simplex                                 | convex
`IndSOC`        | Second-order cone                       | convex
`IndSphereL2`   | Euclidean sphere                        | nonconvex
`IndZero`       | Indicator of the zero singleton         | convex

### Norms, pseudo-norms, regularization functions

Name            | Description                         | Properties
----------------|-------------------------------------|----------------
`ElasticNet`    | Elastic-net regularization          | convex
`NormL0`        | L0 pseudo-norm                      | nonconvex
`NormL1`        | L1 norm                             | convex
`NormL2`        | Euclidean norm                      | convex
`NormL21`       | Sum-of-L2 norms                     | convex
`NormLinf`      | L-infinity norm                     | convex
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
