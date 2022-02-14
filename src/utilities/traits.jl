is_prox_accurate(::Type) = true
is_prox_accurate(::T) where T = is_prox_accurate(T)

is_separable(::Type) = false
is_separable(::T) where T = is_separable(T)

is_singleton(::Type) = false
is_singleton(::T) where T = is_singleton(T)

is_cone(::Type) = false
is_cone(::T) where T = is_cone(T)

is_affine(T::Type) = is_singleton(T)
is_affine(::T) where T = is_affine(T)

is_set(T::Type) = is_cone(T) || is_affine(T)
is_set(::T) where T = is_set(T)

is_positively_homogeneous(T::Type) = is_cone(T)
is_positively_homogeneous(::T) where T = is_positively_homogeneous(T)

is_support(T::Type) = is_convex(T) && is_positively_homogeneous(T)
is_support(::T) where T = is_support(T)

is_smooth(::Type) = false
is_smooth(::T) where T = is_smooth(T)

is_quadratic(T::Type) = is_generalized_quadratic(T) && is_smooth(T)
is_quadratic(::T) where T = is_quadratic(T)

is_strongly_convex(::Type) = false
is_strongly_convex(::T) where T = is_strongly_convex(T)
