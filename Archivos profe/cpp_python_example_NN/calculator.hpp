#pragma once
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace calculator {
    // Scalar operations
    double add(double a, double b);
    double subtract(double a, double b);
    double multiply(double a, double b);
    double divide(double a, double b);

    // Matrix operations
    py::array_t<double> matrix_add(py::array_t<double> a, py::array_t<double> b);
    py::array_t<double> matrix_subtract(py::array_t<double> a, py::array_t<double> b);
    py::array_t<double> matrix_multiply(py::array_t<double> a, py::array_t<double> b);
} 