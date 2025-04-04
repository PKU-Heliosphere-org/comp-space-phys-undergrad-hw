from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os

class get_pybind_include:
    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

ext_modules = [
    Extension(
        'solar_wind_cpp_core',
        ['solar_wind_cpp_core.cpp'],
        include_dirs=[
            get_pybind_include(),
            get_pybind_include(user=True)
        ],
        language='c++',
        extra_compile_args=['-std=c++14', '-O3'],
        extra_link_args=[],
    ),
]

setup(
    name='solar_wind_cpp_core',
    version='0.1',
    author='Author',
    author_email='author@example.com',
    description='C++ implementation of solar wind simulation core functions',
    ext_modules=ext_modules,
    zip_safe=False,
)