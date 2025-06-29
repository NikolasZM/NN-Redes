import numpy as np
import calculator

def test_matrix_operations():
    # Create sample matrices
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[5.0, 6.0], [7.0, 8.0]])

    # Test matrix addition
    result_add = calculator.matrix_add(a, b)
    expected_add = np.array([[6.0, 8.0], [10.0, 12.0]])
    assert np.allclose(result_add, expected_add)
    print("Matrix addition test passed")

    # Test matrix subtraction
    result_sub = calculator.matrix_subtract(a, b)
    expected_sub = np.array([[-4.0, -4.0], [-4.0, -4.0]])
    assert np.allclose(result_sub, expected_sub)
    print("Matrix subtraction test passed")

    # Test matrix multiplication
    result_mul = calculator.matrix_multiply(a, b)
    expected_mul = np.array([[19.0, 22.0], [43.0, 50.0]])
    assert np.allclose(result_mul, expected_mul)
    print("Matrix multiplication test passed")

if __name__ == "__main__":
    test_matrix_operations() 