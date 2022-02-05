# squared Euclidean distance from a set

export SqrDistL2

"""
**Squared distance from a set**

    SqrDistL2(ind_S, λ=1)

Given `ind_S` the indicator function of a set ``S``, and an optional positive parameter `λ`, returns the (weighted) squared Euclidean distance from ``S``, that is function
```math
g(x) = \\tfrac{λ}{2}\\mathrm{dist}_S^2(x) = \\min \\left\\{ \\tfrac{λ}{2}\\|y - x\\|^2 : y \\in S \\right\\}.
```
"""
SqrDistL2(ind, lambda=1) = Postcompose(MoreauEnvelope(ind), lambda)
