from pathlib import Path

from setuptools import setup
import os
import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext

user_path = os.path.expanduser("~")
eigen_include_dir = Path(user_path) / "eigen-3.4.0"


ext_modules = [
    Pybind11Extension(
        "engines",
        ["engines.cpp"],
        include_dirs=[pybind11.get_include(), pybind11.get_include(user=True), eigen_include_dir],
        extra_compile_args=['-O3', '-ffast-math', '-fopenmp'],  # 添加 '-fopenmp' 以启用 OpenMP 支持
        extra_link_args=['-fopenmp'],  # 链接时也添加 '-fopenmp'
    ),
]

setup(
    name="engines",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
