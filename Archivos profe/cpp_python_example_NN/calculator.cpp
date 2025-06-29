#include "calculator.hpp"
#include <stdexcept>

namespace calculator {
    // Scalar operations
    double add(double a, double b) {
        return a + b;
    }

    double subtract(double a, double b) {
        return a - b;
    }

    double multiply(double a, double b) {
        return a * b;
    }

    double divide(double a, double b) {
        if (b == 0) throw std::runtime_error("Division by zero");
        return a / b;
    }

    // Matrix operations
    py::array_t<double> matrix_add(py::array_t<double> a, py::array_t<double> b) {
        py::buffer_info buf1 = a.request(), buf2 = b.request();
        
        if (buf1.ndim != 2 || buf2.ndim != 2)
            throw std::runtime_error("Number of dimensions must be 2");
        
        if (buf1.shape[0] != buf2.shape[0] || buf1.shape[1] != buf2.shape[1])
            throw std::runtime_error("Input shapes must match");

        auto result = py::array_t<double>(buf1.shape);
        py::buffer_info buf3 = result.request();

        double *ptr1 = static_cast<double *>(buf1.ptr);
        double *ptr2 = static_cast<double *>(buf2.ptr);
        double *ptr3 = static_cast<double *>(buf3.ptr);

        for (size_t i = 0; i < buf1.shape[0] * buf1.shape[1]; i++) {
            ptr3[i] = ptr1[i] + ptr2[i];
        }

        return result;
    }

    py::array_t<double> matrix_subtract(py::array_t<double> a, py::array_t<double> b) {
        py::buffer_info buf1 = a.request(), buf2 = b.request();
        
        if (buf1.ndim != 2 || buf2.ndim != 2)
            throw std::runtime_error("Number of dimensions must be 2");
        
        if (buf1.shape[0] != buf2.shape[0] || buf1.shape[1] != buf2.shape[1])
            throw std::runtime_error("Input shapes must match");

        auto result = py::array_t<double>(buf1.shape);
        py::buffer_info buf3 = result.request();

        double *ptr1 = static_cast<double *>(buf1.ptr);
        double *ptr2 = static_cast<double *>(buf2.ptr);
        double *ptr3 = static_cast<double *>(buf3.ptr);

        for (size_t i = 0; i < buf1.shape[0] * buf1.shape[1]; i++) {
            ptr3[i] = ptr1[i] - ptr2[i];
        }

        return result;
    }

    py::array_t<double> matrix_multiply(py::array_t<double> a, py::array_t<double> b) {
        py::buffer_info buf1 = a.request(), buf2 = b.request();
        
        if (buf1.ndim != 2 || buf2.ndim != 2)
            throw std::runtime_error("Number of dimensions must be 2");
        
        if (buf1.shape[1] != buf2.shape[0])
            throw std::runtime_error("Input shapes are not compatible for matrix multiplication");

        auto result = py::array_t<double>({buf1.shape[0], buf2.shape[1]});
        py::buffer_info buf3 = result.request();

        double *ptr1 = static_cast<double *>(buf1.ptr);
        double *ptr2 = static_cast<double *>(buf2.ptr);
        double *ptr3 = static_cast<double *>(buf3.ptr);

        for (size_t i = 0; i < buf1.shape[0]; i++) {
            for (size_t j = 0; j < buf2.shape[1]; j++) {
                double sum = 0;
                for (size_t k = 0; k < buf1.shape[1]; k++) {
                    sum += ptr1[i * buf1.shape[1] + k] * ptr2[k * buf2.shape[1] + j];
                }
                ptr3[i * buf2.shape[1] + j] = sum;
            }
        }

        return result;
    }
} 