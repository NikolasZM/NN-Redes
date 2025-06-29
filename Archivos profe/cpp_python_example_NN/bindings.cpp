#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "calculator.hpp"

namespace py = pybind11;

PYBIND11_MODULE(calculator, m) {
    m.doc() = "Calculator module implemented in C++"; // Module docstring

    // Scalar operations
    m.def("add", &calculator::add, "Add two numbers");
    m.def("subtract", &calculator::subtract, "Subtract two numbers");
    m.def("multiply", &calculator::multiply, "Multiply two numbers");
    m.def("divide", &calculator::divide, "Divide two numbers");

    // Matrix operations
    m.def("matrix_add", &calculator::matrix_add, "Add two matrices",
          py::arg("a").noconvert(), py::arg("b").noconvert());
    m.def("matrix_subtract", &calculator::matrix_subtract, "Subtract two matrices",
          py::arg("a").noconvert(), py::arg("b").noconvert());
    m.def("matrix_multiply", &calculator::matrix_multiply, "Multiply two matrices",
          py::arg("a").noconvert(), py::arg("b").noconvert());
} 