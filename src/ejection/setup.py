from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        'ejection_ext',
        ['ejection_wrapper.cpp'],
        include_dirs=[],
        language='c++'
    ),
]


setup(
    name='ejection',
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
)