# Max function

"""
  Maximum(λ::Real=1.0)

Returns the function `g(x) = λ⋅maximum(x)`, for a nonnegative parameter `λ ⩾ 0`.
"""

Maximum{R <: Real}(lambda::R=1.0) = SumLargest(one(Int32), lambda)
