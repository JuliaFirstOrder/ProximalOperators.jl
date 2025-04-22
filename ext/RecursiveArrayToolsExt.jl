module RecursiveArrayToolsExt
using RecursiveArrayTools
using ProximalOperators
import ProximalCore: prox, prox!, gradient, gradient!

(f::PrecomposedSlicedSeparableSum)(x::ArrayPartition) = f(x.x)
prox!(y::ArrayPartition, f::PrecomposedSlicedSeparableSum, x::ArrayPartition, gamma) = prox!(y.x, f, x.x, gamma)

(g::SeparableSum)(xs::ArrayPartition) = g(xs.x)
prox!(ys::ArrayPartition, g::SeparableSum, xs::ArrayPartition, gamma::Number) = prox!(ys.x, g, xs.x, gamma)
prox!(ys::ArrayPartition, g::SeparableSum, xs::ArrayPartition, gammas::Tuple) = prox!(ys.x, g, xs.x, gammas)
function prox(g::SeparableSum, xs::ArrayPartition, gamma=1)
    y, fy = prox(g, xs.x, gamma)
    return ArrayPartition(y), fy
end
gradient!(grads::ArrayPartition, g::SeparableSum, xs::ArrayPartition) = gradient!(grads.x, g, xs.x)
gradient(g::SeparableSum, xs::ArrayPartition) = gradient(g, xs.x)

end # module RecursiveArrayToolsExt
