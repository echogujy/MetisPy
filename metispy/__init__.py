"""
MetisPy: PyTorch-compatible bindings for METIS graph partitioning
"""

from .metis import partition_graph, node_nd, partition_mesh_nodal, partition_mesh_dual

__version__ = '0.1.0'
__all__ = ['partition_graph', 'node_nd', 'partition_mesh_nodal', 'partition_mesh_dual']
