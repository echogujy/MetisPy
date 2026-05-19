"""
MetisPy: PyTorch-compatible bindings for METIS graph partitioning
"""

from .metis import partition_graph

__version__ = '0.1.0'
__all__ = ['partition_graph']
