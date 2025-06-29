from setuptools import setup, Extension
import pybind11
import numpy as np

ext_modules = [
    Extension(
        "calculator",
        ["calculator.cpp", "bindings.cpp"],
        include_dirs=[pybind11.get_include(), np.get_include()],
        language='c++',
        extra_compile_args=['-std=c++11']
    ),
]

setup(
    name="calculator",
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.10.0', 'numpy>=1.20.0'],
    python_requires=">=3.6",
) 