"""
Basic usage example for MetisPy
"""

import numpy as np
from metispy import partition_graph, node_nd, partition_mesh_nodal, partition_mesh_dual


def example_basic_partition():
    """Basic graph partitioning example"""
    print("Example 1: Basic graph partitioning")
    
    # Create a simple 3-vertex graph in CSR format
    # Graph: 0 -- 1 -- 2 (with 0 also connected to 2)
    xadj = np.array([0, 2, 4, 6], dtype=np.int64)
    adjncy = np.array([1, 2, 0, 2, 0, 1], dtype=np.int64)
    
    # Partition into 2 parts using K-way method
    partition = partition_graph(
        xadj=xadj,
        adjncy=adjncy,
        nparts=2,
        method='kway'
    )
    
    print(f"Partition result: {partition}")
    print(f"Vertex 0 -> partition {partition[0]}")
    print(f"Vertex 1 -> partition {partition[1]}")
    print(f"Vertex 2 -> partition {partition[2]}")
    print()


def example_with_vertex_weights():
    """Example with vertex weights"""
    print("Example 2: Partitioning with vertex weights")
    
    # Create a graph with 4 vertices
    xadj = np.array([0, 2, 4, 6, 8], dtype=np.int64)
    adjncy = np.array([1, 2, 0, 2, 0, 3, 2, 1], dtype=np.int64)
    
    # Vertex weights (e.g., computational cost)
    vwgt = np.array([1, 2, 1, 3], dtype=np.int64)
    
    # Partition into 2 parts
    partition = partition_graph(
        xadj=xadj,
        adjncy=adjncy,
        vwgt=vwgt,
        nparts=2,
        method='kway'
    )
    
    print(f"Vertex weights: {vwgt}")
    print(f"Partition result: {partition}")
    print()


def example_with_edge_weights():
    """Example with edge weights"""
    print("Example 3: Partitioning with edge weights")
    
    # Create a graph
    xadj = np.array([0, 2, 4, 6], dtype=np.int64)
    adjncy = np.array([1, 2, 0, 2, 0, 1], dtype=np.int64)
    
    # Edge weights (e.g., communication cost)
    adjwgt = np.array([5, 1, 5, 1, 1, 1], dtype=np.int64)
    
    # Partition into 2 parts
    partition = partition_graph(
        xadj=xadj,
        adjncy=adjncy,
        adjwgt=adjwgt,
        nparts=2,
        method='kway'
    )
    
    print(f"Edge weights: {adjwgt}")
    print(f"Partition result: {partition}")
    print()


def example_recursive_method():
    """Example using recursive partitioning"""
    print("Example 4: Recursive partitioning method")
    
    # Create a larger graph
    xadj = np.array([0, 3, 6, 9, 12], dtype=np.int64)
    adjncy = np.array([1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2], dtype=np.int64)
    
    # Partition using recursive method
    partition = partition_graph(
        xadj=xadj,
        adjncy=adjncy,
        nparts=2,
        method='recursive'
    )
    
    print(f"Recursive partition result: {partition}")
    print()


def example_multiple_partitions():
    """Example partitioning into multiple parts"""
    print("Example 5: Partitioning into 4 parts")
    
    # Create a graph with 8 vertices
    xadj = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16], dtype=np.int64)
    adjncy = np.array([1, 2, 0, 2, 0, 1, 3, 4, 2, 5, 4, 6, 5, 7, 6, 7], dtype=np.int64)
    
    # Partition into 4 parts
    partition = partition_graph(
        xadj=xadj,
        adjncy=adjncy,
        nparts=4,
        method='kway'
    )
    
    print(f"Partition into 4 parts: {partition}")
    print()


def example_matrix_reordering():
    """Example showing sparse matrix fill-reducing reordering"""
    print("Example 6: Sparse matrix fill-reducing reordering (NodeND)")
    
    # 3-vertex line graph structure
    xadj = np.array([0, 2, 4, 6], dtype=np.int64)
    adjncy = np.array([1, 2, 0, 2, 0, 1], dtype=np.int64)
    
    perm, iperm = node_nd(xadj, adjncy)
    
    print(f"Permutation array (perm): {perm}")
    print(f"Inverse permutation array (iperm): {iperm}")
    print()


def example_mesh_partitioning():
    """Example showing mesh partitioning (Nodal & Dual)"""
    print("Example 7: Mesh Partitioning")
    
    # Two quad elements sharing an edge (nodes 1 and 2)
    # Element 0: Nodes 0, 1, 2, 3
    # Element 1: Nodes 1, 4, 5, 2
    ne = 2
    nn = 6
    eptr = np.array([0, 4, 8], dtype=np.int64)
    eind = np.array([0, 1, 2, 3, 1, 4, 5, 2], dtype=np.int64)
    
    # 1. Nodal mesh partitioning
    objval_nodal, epart_nodal, npart_nodal = partition_mesh_nodal(
        ne=ne, nn=nn, eptr=eptr, eind=eind, nparts=2
    )
    print(f"Nodal mesh partition:")
    print(f"  Objective value: {objval_nodal}")
    print(f"  Element partitions: {epart_nodal}")
    print(f"  Node partitions:    {npart_nodal}")
    
    # 2. Dual mesh partitioning (requiring 2 common nodes to share an edge)
    objval_dual, epart_dual, npart_dual = partition_mesh_dual(
        ne=ne, nn=nn, eptr=eptr, eind=eind, ncommon=2, nparts=2
    )
    print(f"Dual mesh partition:")
    print(f"  Objective value: {objval_dual}")
    print(f"  Element partitions: {epart_dual}")
    print(f"  Node partitions:    {npart_dual}")
    print()


if __name__ == "__main__":
    try:
        example_basic_partition()
        example_with_vertex_weights()
        example_with_edge_weights()
        example_recursive_method()
        example_multiple_partitions()
        example_matrix_reordering()
        example_mesh_partitioning()
        print("All examples completed successfully!")
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure MetisPy is properly installed.")

