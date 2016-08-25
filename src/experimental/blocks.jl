# Definitions useful to work with tuples of variable blocks.
# First we extend sum, subtraction, to (omogeneous) tuples.

import Base.+, Base.-

function +(a::Tuple, b::Tuple)
   N = length(a)
   if length(b) != N error("operands must have the same length") end
   return map(i -> a[i]+b[i], (1:N...))
end

function -(a::Tuple, b::Tuple)
   N = length(a)
   if length(b) != N error("operands must have the same length") end
   return map(i -> a[i]-b[i], (1:N...))
end

# Then we would need a way to stack linear operators horizontally/vertically
# pretty much like they do in LinearOperators, but they only handle operators
# from/to spaces of Array{T,1}.
