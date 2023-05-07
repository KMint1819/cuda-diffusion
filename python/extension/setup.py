from setuptools import setup, Extension, find_packages
from torch.utils import cpp_extension

# think of a name for your extension module using words (aqua, vincent, john)
setup(name='torda',
      ext_modules=[cpp_extension.CppExtension('torda', [
          'src/torda.cc',
      ])],
      packages=find_packages(),
      cmdclass={'build_ext': cpp_extension.BuildExtension})