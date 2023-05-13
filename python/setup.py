from setuptools import setup, Extension, find_packages
from torch.utils import cpp_extension

setup(name='gten',
      ext_modules=[cpp_extension.CppExtension('gten_backend', [
          'crossattn/cpp/gten.cc',
          'crossattn/cpp/gten_cuda.cu',
      ])],
      py_modules=['gten'],
      packages=find_packages(),
      cmdclass={'build_ext': cpp_extension.BuildExtension})