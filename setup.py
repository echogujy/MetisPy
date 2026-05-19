from setuptools import setup, find_packages
from setuptools.extension import Extension
import os
import sys
import pybind11

# Check if METIS is built
project_dir = os.path.dirname(os.path.abspath(__file__))
metis_dist_dir = os.path.join(project_dir, 'third_party', 'METIS', 'build', 'dist')
gklib_dist_dir = os.path.join(project_dir, 'third_party', 'GKlib', 'dist')
# Check if METIS library exists in dist directory
if os.path.exists(os.path.join(metis_dist_dir, 'lib', 'libmetis.a')):
    metis_lib_dir = os.path.join(metis_dist_dir, 'lib')
    metis_include_dir = os.path.join(metis_dist_dir, 'include')
    gklib_lib_dir = os.path.join(gklib_dist_dir, 'lib')
    gklib_include_dir = os.path.join(gklib_dist_dir, 'include')

else:
    print("Warning: METIS library not found.")
    print("Please build METIS first by running:")
    print("  cd third_party/GKlib && mkdir build && cd build && cmake -DCMAKE_INSTALL_PREFIX=../dist .. && make install")
    print("  cd ../../METIS && make config gklib_path=../GKlib/dist prefix=dist i64=1 && make install")
    print("Or set METIS_LIB_DIR and METIS_INCLUDE_DIR environment variables.")
    metis_lib_dir = os.environ.get('METIS_LIB_DIR', '')
    metis_include_dir = os.environ.get('METIS_INCLUDE_DIR', '')

setup(
    name='metispy',
    version='0.1.0',
    author='MetisPy Contributors',
    description='PyTorch-compatible bindings for METIS graph partitioning',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/MetisPy',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.19.0',
        'pybind11>=2.6.0',
    ],
    setup_requires=[
        'pybind11>=2.6.0',
    ],
    ext_modules=[
        Extension(
            name='metispy._metis',
            sources=[
                'csrc/metis_binding.cpp',
            ],
            include_dirs=[
                metis_include_dir,
                gklib_include_dir,
                pybind11.get_include(),              
                pybind11.get_include(user=True),    
            ],
            extra_objects=[
                os.path.join(metis_lib_dir, "libmetis.a"),
                os.path.join(gklib_lib_dir, "libGKlib.a"),
            ],
            extra_compile_args=['-O3', '-std=c++14'],
            language='c++',
        )
    ],
    zip_safe=False,
)
