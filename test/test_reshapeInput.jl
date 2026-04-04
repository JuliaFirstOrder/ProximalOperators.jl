using LinearAlgebra
using ProximalOperators
using Test

# Define a simple test function that we can use with ReshapeInput
struct SimpleTestFunc end

# Make it callable - returns squared norm
# This function requires 2D input (matrix), and will error for vectors or higher-dimensional arrays
function (::SimpleTestFunc)(x)
    if ndims(x) != 2
        throw(DimensionMismatch("SimpleTestFunc requires 2D input (matrix), got $(ndims(x))D array"))
    end
    return sum(abs2, x)
end

# Define a prox! method for SimpleTestFunc
function ProximalOperators.prox!(y, f::SimpleTestFunc, x, gamma)
    if ndims(x) != 2
        throw(DimensionMismatch("SimpleTestFunc requires 2D input (matrix), got $(ndims(x))D array"))
    end
    # Simple soft-thresholding prox: prox(||·||^2) = x / (1 + 2*gamma)
    y .= x ./ (1 + 2 * gamma)
    return sum(abs2, y)
end

# Define a gradient! method for SimpleTestFunc
function ProximalOperators.gradient!(y, f::SimpleTestFunc, x)
    if ndims(x) != 2
        throw(DimensionMismatch("SimpleTestFunc requires 2D input (matrix), got $(ndims(x))D array"))
    end
    # Gradient of squared norm: 2*x
    y .= 2 .* x
    return sum(abs2, y)
end



@testset "ReshapeInput Tests" begin
    
    @testset "Basic Function Call with Correct Shape" begin
        # Create a ReshapeInput wrapper
        f = ReshapeInput(SimpleTestFunc(), (2, 2))
        
        # Create input with correct shape
        x = reshape(1.0:4.0, 2, 2)
        result = f(x)
        
        # Should return squared norm of all elements: 1 + 4 + 9 + 16 = 30
        expected = sum(abs2, x)
        @test result ≈ expected
    end
    
    @testset "Function Call with Shape Reshaping" begin
        # Create a ReshapeInput wrapper expecting (2, 2)
        f = ReshapeInput(SimpleTestFunc(), (2, 2))
        
        # Create input as a vector (different shape)
        x = vec(reshape(1.0:4.0, 2, 2))  # [1, 2, 3, 4]
        result = f(x)
        
        # Should reshape to (2, 2) internally and compute squared norm
        x_reshaped = reshape(x, 2, 2)
        expected = sum(abs2, x_reshaped)
        @test result ≈ expected
    end
    
    @testset "Function Call with Multiple Reshaping" begin
        # Create a ReshapeInput wrapper expecting (3, 4)
        f = ReshapeInput(SimpleTestFunc(), (3, 4))
        
        # Create input as a vector of 12 elements
        x = collect(1.0:12.0)
        result = f(x)
        
        # Should reshape to (3, 4) and compute squared norm
        x_reshaped = reshape(x, 3, 4)
        expected = sum(abs2, x_reshaped)
        @test result ≈ expected
    end
    
    @testset "prox! with Correct Shape" begin
        # Create a ReshapeInput wrapper
        f = ReshapeInput(SimpleTestFunc(), (2, 2))
        
        # Create input and output with correct shape
        x = reshape(1.0:4.0, 2, 2)
        y = zeros(2, 2)
        gamma = 0.5
        
        result = prox!(y, f, x, gamma)
        
        # prox of squared norm with soft-thresholding
        expected_y = x ./ (1 + 2 * gamma)
        expected_result = sum(abs2, expected_y)
        
        @test y ≈ expected_y
        @test result ≈ expected_result
    end
    
    @testset "prox! with Shape Reshaping" begin
        # Create a ReshapeInput wrapper expecting (2, 2)
        f = ReshapeInput(SimpleTestFunc(), (2, 2))
        
        # Create input and output as vectors
        x = collect(1.0:4.0)
        y = zeros(4)
        gamma = 0.5
        
        result = prox!(y, f, x, gamma)
        
        # Should internally reshape to (2, 2)
        x_reshaped = reshape(x, 2, 2)
        expected_y_reshaped = x_reshaped ./ (1 + 2 * gamma)
        expected_result = sum(abs2, expected_y_reshaped)
        
        # y should contain the reshaped result flattened back
        y_expected = vec(expected_y_reshaped)
        @test y ≈ y_expected
        @test result ≈ expected_result
    end
    
    @testset "gradient! with Correct Shape" begin
        # Create a ReshapeInput wrapper
        f = ReshapeInput(SimpleTestFunc(), (2, 2))
        
        # Create input and output with correct shape
        x = reshape(1.0:4.0, 2, 2)
        y = zeros(2, 2)
        
        result = gradient!(y, f, x)
        
        # Gradient of squared norm: 2*x
        expected_y = 2 .* x
        expected_result = sum(abs2, expected_y)
        
        @test y ≈ expected_y
        @test result ≈ expected_result
    end
    
    @testset "gradient! with Shape Reshaping" begin
        # Create a ReshapeInput wrapper expecting (2, 2)
        f = ReshapeInput(SimpleTestFunc(), (2, 2))
        
        # Create input and output as vectors
        x = collect(1.0:4.0)
        y = zeros(4)
        
        result = gradient!(y, f, x)
        
        # Should internally reshape to (2, 2)
        x_reshaped = reshape(x, 2, 2)
        expected_y_reshaped = 2 .* x_reshaped
        expected_result = sum(abs2, expected_y_reshaped)
        
        # y should contain the reshaped result flattened back
        y_expected = vec(expected_y_reshaped)
        @test y ≈ y_expected
        @test result ≈ expected_result
    end

end
