## TODO

* Update docstrings.
* Add more tests, especially with `Array{Complex}` variables.
* Functions that take either scalar or `Array` parameters when constructing them: check their implementations.
* Get rid of `Array` in favor of `AbstractArray`

## Functions to add

* Nuclear norm.
* Indicator of L1 norm ball.
* Least squares penalty.

## Questions

* Are the typealias `RealArray`, `RealOrComplexArray` and so on OK?
* Can we get rid of `Float64` everywhere in favor of `Real` somehow?
* In `prox!`, is it safe to use `@inbounds`? Because we take the iterator for `x`
and not for `y`, in principle one could provide `y` of the wrong dimension.
