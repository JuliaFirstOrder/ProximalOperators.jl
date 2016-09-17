## TODO

* Make sure to allocate the least amount of memory in call and `prox!`.
  * `IndBox` (tricky, many cases)
  * `IndBallRank` (I'm wondering whether anything smart can be done here)
* Add more tests, especially with `Array{Complex}` variables.
* Functions that take either scalar or `Array` parameters, check their implementations.

## Functions to add

* Nuclear norm.
* Indicator of L1 norm ball.
* Least squares penalty.

## Questions

* Are the typealias `RealArray`, `RealOrComplexArray` and so on OK?
* In `prox!`, is it safe to use `@inbounds`? Because we take the iterator for `x`
and not for `y`, in principle one could provide `y` of the wrong dimension.
