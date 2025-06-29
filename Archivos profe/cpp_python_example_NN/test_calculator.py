import calculator

def test_calculator():
    # Test addition
    result = calculator.add(5.0, 3.0)
    print(f"5 + 3 = {result}")

    # Test subtraction
    result = calculator.subtract(10.0, 4.0)
    print(f"10 - 4 = {result}")

    # Test multiplication
    result = calculator.multiply(6.0, 7.0)
    print(f"6 * 7 = {result}")

    # Test division
    result = calculator.divide(15.0, 3.0)
    print(f"15 / 3 = {result}")

    # Test division by zero exception
    try:
        calculator.divide(10.0, 0.0)
    except Exception as e:
        print("Successfully caught division by zero error")

if __name__ == "__main__":
    test_calculator() 