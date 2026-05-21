# MetisPy

NumPy-compatible bindings for METIS graph partitioning library.

## Overview

MetisPy provides a Python interface to the METIS graph partitioning library using NumPy arrays. This library allows you to partition graphs efficiently using METIS algorithms directly from NumPy. 

## Features

- **NumPy Array Support**: Native support for NumPy arrays as input/output.
- **Graph Partitioning**: Support for K-way and Recursive graph partitioning with vertex/edge weights.
- **Matrix Reordering**: Compute fill-reducing orderings using nested dissection (`node_nd`) for sparse matrices.
- **Mesh Partitioning**: Support for nodal (`partition_mesh_nodal`) and dual (`partition_mesh_dual`) mesh partitioning.
- **Memory Safe**: Direct framework-native memory allocation with NumPy, eliminating dangling pointers and unnecessary copies.

## Installation

### Prerequisites

- Python >= 3.8
- NumPy >= 1.19.0
- PyBind11 >= 2.6.0
- CMake >= 3.10
- C compiler (GCC or Clang)
- pytest [optional]

### Build from source

```bash
git clone https://github.com/echogujy/MetisPy
cd MetisPy
git submodule update --init --recursive

# Build GKlib and METIS dependency first
# Note for ARM architecture: you need to compile GKlib with -DNO_X86=1, i.e.:
# cmake -DCMAKE_INSTALL_PREFIX=../dist -DNO_X86=1 ..
cd third_party/GKlib && mkdir -p build && cd build && cmake -DCMAKE_INSTALL_PREFIX=../dist .. && make install
cd ../../METIS && make config gklib_path=../GKlib/dist prefix=dist i64=1 && make install
cd ../../

# Install MetisPy extension
pip install -e . --no-build-isolation
```

## Quick Start

### 1. Graph Partitioning
```python
import numpy as np
from metispy import partition_graph

# 3-vertex graph: 0 -- 1 -- 2
xadj = np.array([0, 2, 4, 6], dtype=np.int64)
adjncy = np.array([1, 2, 0, 2, 0, 1], dtype=np.int64)

# Partition into 2 parts
parts = partition_graph(xadj, adjncy, nparts=2)
print("Partitions:", parts)
```

### 2. Sparse Matrix Reordering (NodeND)
```python
from metispy import node_nd

# Get permutation vectors for fill-in reduction
perm, iperm = node_nd(xadj, adjncy)
print("Permutation:", perm)
```

### 3. Mesh Partitioning
```python
from metispy import partition_mesh_nodal, partition_mesh_dual

# Two quad elements sharing an edge
eptr = np.array([0, 4, 8], dtype=np.int64)
eind = np.array([0, 1, 2, 3, 1, 4, 5, 2], dtype=np.int64)

# Partition mesh elements and nodes into 2 parts
objval, epart, npart = partition_mesh_nodal(ne=2, nn=6, eptr=eptr, eind=eind, nparts=2)
print("Element partitions:", epart)
print("Node partitions:", npart)
```

## Running Tests & Examples

To run the example script:
```bash
python tests/basic_usage.py
```

To run the test suite:
```bash
python -m pytest tests/test_metis.py
```
