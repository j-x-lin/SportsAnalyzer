from setuptools import setup
from Cython.Build import cythonize

setup(
    name='image_utils',
    version='0.0.1',
    ext_modules=cythonize("process_image.pyx"),
)