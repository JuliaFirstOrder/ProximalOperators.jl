## Functions

The available constructors are listed in the following tables.
You can access the specific documentation of each of them from the command line
of Julia (try typing in `?NormL1`) to have information on their parameters.

### Indicator functions

Name            | Type of set                             | Properties
----------------|-----------------------------------------|----------------
`IndAffine`     | Affine subspace                         | convex
`IndBallLinf`   | L-infinity norm ball                    | convex, separable
`IndBallL0`     | L0 pseudo-norm ball                     | nonconvex
`IndBallL1`     | L1 norm ball                            | convex
`IndBallL2`     | Euclidean ball                          | convex
`IndBallRank`   | Set of matrices with given rank         | nonconvex
`IndBinary`     | Indicator of a binary set               | nonconvex, separable
`IndBox`        | Box                                     | convex, separable
`IndGraph`      | Indicator of the graph of a linear op.  | convex
`IndExpPrimal`  | Indicator of (primal) exponential cone  | convex cone
`IndExpDual`    | Indicator of (dual) exponential cone    | convex cone
`IndFree`       | Indicator of the free cone              | convex cone, separable
`IndHalfspace`  | Halfspace                               | convex
`IndNonnegative`| Nonnegative orthant                     | convex cone, separable
`IndNonpositive`| Nonpositive orthant                     | convex cone, separable
`IndPoint`      | Indicator of a singleton                | convex, separable
`IndPSD`        | Positive semidefinite cone              | convex cone
`IndSimplex`    | Simplex                                 | convex
`IndSOC`        | Second-order cone                       | convex cone
`IndRotatedSOC` | Rotated second-order cone               | convex cone
`IndSphereL2`   | Euclidean sphere                        | nonconvex
`IndZero`       | Indicator of the zero singleton         | convex cone, separable

### Norms, pseudo-norms, regularization functions

Name            | Description                         | Properties
----------------|-------------------------------------|----------------
`ElasticNet`    | Elastic-net regularization          | convex, separable, gradient
`NormL0`        | L0 pseudo-norm                      | nonconvex
`NormL1`        | L1 norm                             | convex, separable, gradient
`NormL2`        | Euclidean norm                      | convex, gradient
`NormL21`       | Sum-of-L2 norms                     | convex
`NormLinf`      | L-infinity norm                     | convex, gradient
`NuclearNorm`   | Nuclear norm                        | convex
`SqrNormL2`     | Squared Euclidean norm              | convex, separable, gradient

### Penalties and other functions

Name            | Description                         | Properties
----------------|-------------------------------------|-----------------
`HingeLoss`     | Hinge loss function                 | convex, separable, gradient
`HuberLoss`     | Huber loss function                 | convex, gradient
`LogBarrier`    | Logarithmic barrier                 | convex, separable, gradient
`LeastSquares`  | Sum-of-residual-squares             | convex, gradient
`Maximum`       | Maximum coordinate of a vector      | convex
`Linear`        | Linear function                     | convex, separable, gradient
`Quadratic`     | Quadratic function                  | convex, gradient
`SumPositive`   | Sum of the positive coefficients    | convex, gradient

### Distances

Name            | Description                                          | Properties
----------------|------------------------------------------------------|----------------
`DistL2`        | Euclidean distance from a convex set                 | convex, gradient
`SqrDistL2`     | Squared Euclidean distance from a convex set         | convex, gradient
