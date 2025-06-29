# C++ Calculator Module for Python

This is a simple example of how to create a C++ module that can be used in Python using pybind11.

## Requirements

- Python 3.6 or higher
- A C++ compiler (MSVC on Windows, GCC on Linux, or Clang on macOS)
- pybind11
- setuptools

## Installation

1. First, install pybind11:
```bash
pip install pybind11
```

2. Build the module:
```bash
python setup.py build_ext --inplace
```

## Usage

After building the module, you can use it in Python like this:

```python
import calculator

# Add two numbers
result = calculator.add(5.0, 3.0)
print(result)  # Output: 8.0

# Subtract two numbers
result = calculator.subtract(10.0, 4.0)
print(result)  # Output: 6.0

# Multiply two numbers
result = calculator.multiply(6.0, 7.0)
print(result)  # Output: 42.0

# Divide two numbers
result = calculator.divide(15.0, 3.0)
print(result)  # Output: 5.0
```

## Testing

Run the test file to see the calculator in action:
```bash
python test_calculator.py
``` 