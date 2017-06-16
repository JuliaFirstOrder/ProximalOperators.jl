# Calculus rules

The calculus rules described in the following allow to modify and combine [functions](functions.md), to obtain new ones with efficiently computable proximal mapping.

## Duality

```@docs
Conjugate
```

## Distances from convex sets

When the indicator of a convex set is constructed (see [Indicators of sets](@ref)) the (squared) distance from the set can be constructed using the following:

```@docs
DistL2
SqrDistL2
```

## Functions combination

The following means of combination are important in that they allow to represent a very common situation: defining the sum of multiple functions, each applied to an independent block of variables. The following two constructors, [`SeparableSum`](@ref) and [`SlicedSeparableSum`](@ref), allow to do this in two (complementary) ways.

```@docs
SeparableSum
SlicedSeparableSum
```

## Functions regularization

```@docs
MoreauEnvelope
Regularize
```

## Pre- and post-transformations

```@docs
Postcompose
Tilt
Translate
```
