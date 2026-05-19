"""
Python API for METIS graph partitioning with NumPy array support
"""

import numpy as np
from typing import Optional


def partition_graph(
    xadj: np.ndarray,
    adjncy: np.ndarray,
    nparts: int,
    vwgt: Optional[np.ndarray] = None,
    adjwgt: Optional[np.ndarray] = None,
    ncon: int = 1,
    method: str = 'kway'
) -> np.ndarray:
    """
    Partition a graph using METIS.
    
    Parameters:
    -----------
    xadj : np.ndarray
        CSR pointer array, shape (nvtxs + 1,), dtype int64
        xadj[i] to xadj[i+1]-1 gives the range of neighbors for vertex i
    adjncy : np.ndarray
        CSR adjacency array, shape (nnz,), dtype int64
        Contains the actual neighbor indices
    nparts : int
        Number of partitions to create
    vwgt : np.ndarray, optional
        Vertex weights, shape (nvtxs * ncon,), dtype int64
        If None, all vertices have weight 1
    adjwgt : np.ndarray, optional
        Edge weights, shape (nnz,), dtype int64
        If None, all edges have weight 1
    ncon : int, optional
        Number of balancing constraints (default: 1)
    method : str, optional
        Partitioning method: 'kway' or 'recursive' (default: 'kway')
    
    Returns:
    --------
    np.ndarray
        Partition assignment for each vertex, shape (nvtxs,), dtype int64
        Values range from 0 to nparts-1
    
    Raises:
    -------
    ValueError
        If input arrays have incorrect dtype or shape
    RuntimeError
        If METIS partitioning fails
    
    Examples:
    ---------
    >>> import numpy as np
    >>> from metispy import partition_graph
    >>> 
    >>> # Create a simple 3-vertex graph
    >>> xadj = np.array([0, 2, 4, 6], dtype=np.int64)
    >>> adjncy = np.array([1, 2, 0, 2, 0, 1], dtype=np.int64)
    >>> 
    >>> # Partition into 2 parts
    >>> partition = partition_graph(xadj, adjncy, nparts=2)
    >>> print(partition)
    """
    # Import the C++ extension
    from metispy import _metis
    # try:
    #     from metispy import _metis
    # except ImportError:
    #     raise ImportError(
    #         "METIS C++ extension not found. "
    #         "Please install MetisPy with: pip install -e ."
    #     )
    
    # Validate inputs
    if not isinstance(xadj, np.ndarray):
        raise TypeError("xadj must be a numpy array")
    if not isinstance(adjncy, np.ndarray):
        raise TypeError("adjncy must be a numpy array")
    
    if xadj.dtype != np.int64:
        raise ValueError("xadj must have dtype int64")
    if adjncy.dtype != np.int64:
        raise ValueError("adjncy must have dtype int64")
    
    if xadj.ndim != 1:
        raise ValueError("xadj must be 1-dimensional")
    if adjncy.ndim != 1:
        raise ValueError("adjncy must be 1-dimensional")
    
    if vwgt is not None:
        if not isinstance(vwgt, np.ndarray):
            raise TypeError("vwgt must be a numpy array")
        if vwgt.dtype != np.int64:
            raise ValueError("vwgt must have dtype int64")
        if vwgt.ndim != 1:
            raise ValueError("vwgt must be 1-dimensional")
    
    if adjwgt is not None:
        if not isinstance(adjwgt, np.ndarray):
            raise TypeError("adjwgt must be a numpy array")
        if adjwgt.dtype != np.int64:
            raise ValueError("adjwgt must have dtype int64")
        if adjwgt.ndim != 1:
            raise ValueError("adjwgt must be 1-dimensional")
    
    if method not in ['kway', 'recursive']:
        raise ValueError("method must be 'kway' or 'recursive'")
    
    if nparts <= 0:
        raise ValueError("nparts must be positive")
    
    # Ensure arrays are contiguous
    xadj = np.ascontiguousarray(xadj)
    adjncy = np.ascontiguousarray(adjncy)
    
    vwgt_np = np.ascontiguousarray(vwgt) if vwgt is not None else np.array([], dtype=np.int64)
    adjwgt_np = np.ascontiguousarray(adjwgt) if adjwgt is not None else np.array([], dtype=np.int64)
    
    # Call the C++ extension
    partition = _metis.partition(
        xadj,
        adjncy,
        nparts,
        vwgt_np,
        adjwgt_np,
        ncon,
        method
    )
    
    return partition
