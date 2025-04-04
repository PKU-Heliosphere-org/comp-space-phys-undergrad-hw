from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os
import subprocess

class get_pybind_include:
    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

def is_gcc_available():
    try:
        subprocess.run(['g++', '--version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False

use_openmp = is_gcc_available()

if use_openmp:
    print("Using GCC with OpenMP support")
    os.environ['CC'] = 'gcc'
    os.environ['CXX'] = 'g++'
    compile_args = ['-std=c++14', '-O3', '-fopenmp']
    link_args = ['-fopenmp']
else:
    print("Using default compiler without OpenMP support")
    compile_args = ['-std=c++14', '-O3']
    link_args = []

ext_modules = [
    Extension(
        'solar_wind_cpp_core',
        ['solar_wind_cpp_core.cpp'],
        include_dirs=[
            get_pybind_include(),
            get_pybind_include(user=True)
        ],
        language='c++',
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    ),
]

class BuildExt(build_ext):
    def build_extensions(self):
        if use_openmp:
            self.compiler.compiler_so[0] = 'g++'
            self.compiler.compiler_cxx[0] = 'g++'
            self.compiler.linker_so[0] = 'g++'
        build_ext.build_extensions(self)

setup(
    name='solar_wind_cpp_core',
    version='0.1',
    author='Author',
    author_email='author@example.com',
    description='C++ implementation of solar wind simulation core functions',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
)