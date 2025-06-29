import calculator
import torch

# Basic operations
result = calculator.add(5.0, 3.0)
print(result)  # Output: 8.0

# Tensor operations
matrix1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
matrix2 = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

# Scalar multiplication
result = calculator.multiply_matrix(matrix1, 2.0)
print(result)

# Matrix addition
result = calculator.add_matrices(matrix1, matrix2)
print(result)

# Matrix multiplication
result = calculator.matrix_multiply(matrix1, matrix2)
print(result)

# GPU Support (if available)
if torch.cuda.is_available():
    gpu_matrix = matrix1.cuda()
    result = calculator.multiply_matrix(gpu_matrix, 2.0)
    print(result)  # Result will be on CPU

