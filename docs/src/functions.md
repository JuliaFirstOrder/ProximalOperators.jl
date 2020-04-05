# Functions

Here we list the available functions, grouped by category. Each function is documented with its exact definition and the necessary parameters for construction.
The proximal mapping (and gradient, when defined) of such functions is computed by calling the [`prox`](@ref) and [`prox!`](@ref) methods (and [`gradient`](@ref), [`gradient!`](@ref), when defined).
These functions can be modified and/or combined together to make new ones, by means of [calculus rules](calculus.md).

## Indicators of sets

When function ``f`` is the indicator function of a set ``S``, that is
```math
f(x) = δ_S(x) =
\begin{cases}
0 & \text{if}\ x \in S, \\
+∞ & \text{otherwise},
\end{cases}
```
then ``\mathrm{prox}_{γf} = Π_S`` is the projection onto ``S``.
Therefore ProximalOperators includes in particular projections onto commonly used sets, which are here listed.

```@docs
IndAffine
IndBallLinf   
IndBallL0     
IndBallL1     
IndBallL2     
IndBallRank   
IndBinary
IndBox  
IndGraph     
IndHalfspace  
IndHyperslab
IndPoint
IndPolyhedral              
IndSimplex    
IndSphereL2          
```

## Indicators of convex cones

An important class of sets in optimization is that of convex cones.
These are used in particular for formulating [cone programming problems](https://en.wikipedia.org/wiki/Conic_optimization), a family of problems which includes linear programs (LP), quadratic programs (QP), quadratically constrained quadratic programs (QCQP) and semidefinite programs (SDP).

```@docs
IndExpPrimal
IndExpDual
IndFree
IndNonnegative
IndNonpositive
IndPSD
IndSOC
IndRotatedSOC
IndZero
```

## Norms and regularization functions

```@docs
CubeNormL2
ElasticNet
NormL0
NormL1
NormL2
NormL21
NormLinf
NuclearNorm
SqrNormL2
```

## Penalties and other functions

```@docs
CrossEntropy
HingeLoss
HuberLoss
LeastSquares
Linear
LogBarrier
LogisticLoss
Maximum
Quadratic
SqrHingeLoss
SumPositive
```

## Distances from convex sets

When the indicator of a convex set is constructed (see [Indicators of sets](@ref)) the (squared) distance from the set can be constructed using the following:

```@docs
DistL2
SqrDistL2
```
