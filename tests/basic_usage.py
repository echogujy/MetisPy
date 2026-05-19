"""
Basic usage example for MetisPy
"""

import numpy as np
from metispy import partition_graph


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


if __name__ == "__main__":
    try:
        example_basic_partition()
        example_with_vertex_weights()
        example_with_edge_weights()
        example_recursive_method()
        example_multiple_partitions()
        print("All examples completed successfully!")
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure MetisPy is properly installed.")
