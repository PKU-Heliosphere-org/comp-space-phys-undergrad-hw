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
        'solar_wind_cpp_core_v2',
        ['solar_wind_cpp_core_v2.cpp'],
        include_dirs=[
            get_pybind_include(),
            get_pybind_include(user=True)
        ],
        language='c++',
        extra_compile_args=['-std=c++17', '-O3'],  # 使用C++17标准，高优化级别
        # extra_link_args=[],  # 不需要特殊的链接参数
    ),
]

setup(
    name='solar_wind_cpp_core_v2',
    version='0.2',
    author='Author',
    author_email='author@example.com',
    description='Enhanced C++ implementation of solar wind simulation core functions',
    ext_modules=ext_modules,
    zip_safe=False,
)