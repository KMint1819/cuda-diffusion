from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='torda',
      ext_modules=[cpp_extension.CppExtension('torda', ['lib.cc'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})