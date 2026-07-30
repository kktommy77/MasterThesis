from os import path
import setuptools
import torch

from torch.utils import cpp_extension
"""
def main():
    setuptools.setup(
            name='ext_vecadd',
            version= '1.0.0',
            author= 'Kin',
            author_email= 'my@gmail.com',
            packages= setuptools.find_packages(),
            ext_modules= [cpp_extension.CppExtension(name= 'vecadd_cuda', sources= ['./cuda/vecadd.cpp'],extra_compile_args=['-g'],)],
            cmdclass= {'build_ext': cpp_extension.BuildExtension},

            )
    return"""

def main() -> None:
    setuptools.setup(
        name='ext_compression',
        version='1.0.0',
        author='kktommy',
        author_email='kktommy7@gmail.com',
        packages=setuptools.find_packages(),
        ext_modules=[cpp_extension.CppExtension(
            name='compression_cuda',
            sources=[path.join('cuda', 'compression.cpp')],
            libraries=[
                'compression_kernel',
            ],
            library_dirs=[path.join('.', 'cuda')],
            extra_compile_args=['-g', '-fPIC'],
        )],
        cmdclass={'build_ext': cpp_extension.BuildExtension},
    )
    return
 
if __name__ == '__main__':
    main()

#main()
