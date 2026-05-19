# MetisPy

NumPy-compatible bindings for METIS graph partitioning library.

## Overview

MetisPy provides a Python interface to the METIS graph partitioning library using NumPy arrays. This library allows you to partition graphs efficiently using METIS algorithms directly from NumPy, without needing Cython or manual memory management.

## Features

- **NumPy Array Support**: Native support for NumPy arrays as input/output
- **Multiple Partitioning Methods**: Support for both K-way and Recursive partitioning
- **Flexible Graph Representation**: Support for weighted/unweighted vertices and edges
- **Easy Installation**: Built with PyBind11's C++ extension system
- **Open Source**: Licensed under MIT for easy integration

## Installation

### Prerequisites

- Python >= 3.8
- NumPy >= 1.19.0
- PyBind11 >= 2.6.0
- CMake >= 3.10
- C compiler (GCC or Clang)

### Build from source

```bash
git clone https://github.com/yourusername/MetisPy.git
cd MetisPy
git submodule update --init --recursive
cd third_party/GKlib && mkdir build && cd build && cmake -DCMAKE_INSTALL_PREFIX=../dist .. && make install
cd ../../METIS && make config gklib_path=../GKlib/dist prefix=dist i64=1 && make install
cd ../../
pip install -e .
```
