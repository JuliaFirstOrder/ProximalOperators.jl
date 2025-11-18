# wrap a function to reshape the input

export ReshapeInput

"""
    ReshapeInput(f, expected_shape)

Wrap a function to reshape the input.
It is useful when the function `f` expects a specific shape of the input, but you want to pass it a different shape.

```julia
julia> f = ReshapeInput(IndballRank(5), (10, 10))
ReshapeInput(IndBallRank{Int64}(5), (10, 10))

julia> f(rand(100))
Inf
"""
struct ReshapeInput{F, S}
    f::F
    expected_shape::S
end

function (f::ReshapeInput)(x)
    # Check if the input x has the expected shape
    if size(x) != f.expected_shape
        # Reshape the input to the expected shape
        x = reshape(x, f.expected_shape)
    end
    return f.f(x)
end

function prox!(y, f::ReshapeInput, x, gamma)
    # Check if the input x has the expected shape
    if size(x) != f.expected_shape
        # Reshape the input to the expected shape
        x = reshape(x, f.expected_shape)
        y = reshape(y, f.expected_shape)
    end
    return prox!(y, f.f, x, gamma)
end

function gradient!(y, f::ReshapeInput, x)
    # Check if the input x has the expected shape
    if size(x) != f.expected_shape
        # Reshape the input to the expected shape
        x = reshape(x, f.expected_shape)
        y = reshape(y, f.expected_shape)
    end
    return gradient!(y, f.f, x)
end

function prox_naive(f::ReshapeInput, x, gamma)
    # Check if the input x has the expected shape
    if size(x) != f.expected_shape
        # Reshape the input to the expected shape
        x = reshape(x, f.expected_shape)
    end
    return prox_naive(f.f, x, gamma)
end
