from setuptools import setup, Extension
from torch.utils import cpp_extension

# think of a name for your extension module using words (aqua, vincent, john)
setup(name='torda',
      ext_modules=[cpp_extension.CppExtension('torda', [
          'src/torda.cc',
      ])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})