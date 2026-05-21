"""
Python API for METIS graph partitioning with NumPy array support
"""

import numpy as np
from typing import Optional, Tuple


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


def node_nd(
    xadj: np.ndarray,
    adjncy: np.ndarray,
    vwgt: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute fill-reducing ordering of a sparse matrix/graph using METIS NodeND.
    
    Parameters:
    -----------
    xadj : np.ndarray
        CSR pointer array, shape (nvtxs + 1,), dtype int64
    adjncy : np.ndarray
        CSR adjacency array, shape (nnz,), dtype int64
    vwgt : np.ndarray, optional
        Vertex weights, shape (nvtxs,), dtype int64
        
    Returns:
    --------
    perm : np.ndarray
        Permutation array, shape (nvtxs,), dtype int64
    iperm : np.ndarray
        Inverse permutation array, shape (nvtxs,), dtype int64
    """
    from metispy import _metis
    
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
        
    nvtxs = len(xadj) - 1
    if nvtxs < 0:
        raise ValueError("xadj must have at least size 1")
        
    if vwgt is not None:
        if not isinstance(vwgt, np.ndarray):
            raise TypeError("vwgt must be a numpy array")
        if vwgt.dtype != np.int64:
            raise ValueError("vwgt must have dtype int64")
        if vwgt.ndim != 1:
            raise ValueError("vwgt must be 1-dimensional")
        if len(vwgt) != nvtxs:
            raise ValueError("vwgt size must be equal to number of vertices (nvtxs)")
            
    xadj = np.ascontiguousarray(xadj)
    adjncy = np.ascontiguousarray(adjncy)
    vwgt_np = np.ascontiguousarray(vwgt) if vwgt is not None else np.array([], dtype=np.int64)
    
    return _metis.node_nd(xadj, adjncy, vwgt_np)


def partition_mesh_nodal(
    ne: int,
    nn: int,
    eptr: np.ndarray,
    eind: np.ndarray,
    vwgt: Optional[np.ndarray] = None,
    vsize: Optional[np.ndarray] = None,
    nparts: int = 1
) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Partition a mesh using nodal connectivity (METIS_PartMeshNodal).
    
    Parameters:
    -----------
    ne : int
        Number of elements in the mesh
    nn : int
        Number of nodes in the mesh
    eptr : np.ndarray
        Mesh element pointer array, shape (ne + 1,), dtype int64
    eind : np.ndarray
        Mesh element indices array, shape (eptr[ne],), dtype int64
    vwgt : np.ndarray, optional
        Node weights, shape (nn,), dtype int64
    vsize : np.ndarray, optional
        Node sizes, shape (nn,), dtype int64
    nparts : int
        Number of partitions to create
        
    Returns:
    --------
    objval : int
        The edge-cut or volume objective value of the partition
    epart : np.ndarray
        Partition assignment for each element, shape (ne,), dtype int64
    npart : np.ndarray
        Partition assignment for each node, shape (nn,), dtype int64
    """
    from metispy import _metis
    
    if ne <= 0:
        raise ValueError("ne must be positive")
    if nn <= 0:
        raise ValueError("nn must be positive")
    if nparts <= 0:
        raise ValueError("nparts must be positive")
        
    if not isinstance(eptr, np.ndarray):
        raise TypeError("eptr must be a numpy array")
    if not isinstance(eind, np.ndarray):
        raise TypeError("eind must be a numpy array")
        
    if eptr.dtype != np.int64:
        raise ValueError("eptr must have dtype int64")
    if eind.dtype != np.int64:
        raise ValueError("eind must have dtype int64")
        
    if eptr.ndim != 1:
        raise ValueError("eptr must be 1-dimensional")
    if eind.ndim != 1:
        raise ValueError("eind must be 1-dimensional")
        
    if len(eptr) != ne + 1:
        raise ValueError("eptr size must be equal to ne + 1")
        
    if vwgt is not None:
        if not isinstance(vwgt, np.ndarray):
            raise TypeError("vwgt must be a numpy array")
        if vwgt.dtype != np.int64:
            raise ValueError("vwgt must have dtype int64")
        if vwgt.ndim != 1:
            raise ValueError("vwgt must be 1-dimensional")
        if len(vwgt) != nn:
            raise ValueError("vwgt size must be equal to nn (number of nodes)")
            
    if vsize is not None:
        if not isinstance(vsize, np.ndarray):
            raise TypeError("vsize must be a numpy array")
        if vsize.dtype != np.int64:
            raise ValueError("vsize must have dtype int64")
        if vsize.ndim != 1:
            raise ValueError("vsize must be 1-dimensional")
        if len(vsize) != nn:
            raise ValueError("vsize size must be equal to nn (number of nodes)")
            
    eptr = np.ascontiguousarray(eptr)
    eind = np.ascontiguousarray(eind)
    vwgt_np = np.ascontiguousarray(vwgt) if vwgt is not None else np.array([], dtype=np.int64)
    vsize_np = np.ascontiguousarray(vsize) if vsize is not None else np.array([], dtype=np.int64)
    
    return _metis.part_mesh_nodal(ne, nn, eptr, eind, vwgt_np, vsize_np, nparts)


def partition_mesh_dual(
    ne: int,
    nn: int,
    eptr: np.ndarray,
    eind: np.ndarray,
    vwgt: Optional[np.ndarray] = None,
    vsize: Optional[np.ndarray] = None,
    ncommon: int = 1,
    nparts: int = 1
) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Partition a mesh using dual connectivity (METIS_PartMeshDual).
    
    Parameters:
    -----------
    ne : int
        Number of elements in the mesh
    nn : int
        Number of nodes in the mesh
    eptr : np.ndarray
        Mesh element pointer array, shape (ne + 1,), dtype int64
    eind : np.ndarray
        Mesh element indices array, shape (eptr[ne],), dtype int64
    vwgt : np.ndarray, optional
        Element weights, shape (ne,), dtype int64
    vsize : np.ndarray, optional
        Element sizes, shape (ne,), dtype int64
    ncommon : int
        Number of common nodes required to share an edge in the dual graph
    nparts : int
        Number of partitions to create
        
    Returns:
    --------
    objval : int
        The edge-cut or volume objective value of the partition
    epart : np.ndarray
        Partition assignment for each element, shape (ne,), dtype int64
    npart : np.ndarray
        Partition assignment for each node, shape (nn,), dtype int64
    """
    from metispy import _metis
    
    if ne <= 0:
        raise ValueError("ne must be positive")
    if nn <= 0:
        raise ValueError("nn must be positive")
    if nparts <= 0:
        raise ValueError("nparts must be positive")
    if ncommon <= 0:
        raise ValueError("ncommon must be positive")
        
    if not isinstance(eptr, np.ndarray):
        raise TypeError("eptr must be a numpy array")
    if not isinstance(eind, np.ndarray):
        raise TypeError("eind must be a numpy array")
        
    if eptr.dtype != np.int64:
        raise ValueError("eptr must have dtype int64")
    if eind.dtype != np.int64:
        raise ValueError("eind must have dtype int64")
        
    if eptr.ndim != 1:
        raise ValueError("eptr must be 1-dimensional")
    if eind.ndim != 1:
        raise ValueError("eind must be 1-dimensional")
        
    if len(eptr) != ne + 1:
        raise ValueError("eptr size must be equal to ne + 1")
        
    if vwgt is not None:
        if not isinstance(vwgt, np.ndarray):
            raise TypeError("vwgt must be a numpy array")
        if vwgt.dtype != np.int64:
            raise ValueError("vwgt must have dtype int64")
        if vwgt.ndim != 1:
            raise ValueError("vwgt must be 1-dimensional")
        if len(vwgt) != ne:
            raise ValueError("vwgt size must be equal to ne (number of elements)")
            
    if vsize is not None:
        if not isinstance(vsize, np.ndarray):
            raise TypeError("vsize must be a numpy array")
        if vsize.dtype != np.int64:
            raise ValueError("vsize must have dtype int64")
        if vsize.ndim != 1:
            raise ValueError("vsize must be 1-dimensional")
        if len(vsize) != ne:
            raise ValueError("vsize size must be equal to ne (number of elements)")
            
    eptr = np.ascontiguousarray(eptr)
    eind = np.ascontiguousarray(eind)
    vwgt_np = np.ascontiguousarray(vwgt) if vwgt is not None else np.array([], dtype=np.int64)
    vsize_np = np.ascontiguousarray(vsize) if vsize is not None else np.array([], dtype=np.int64)
    
    return _metis.part_mesh_dual(ne, nn, eptr, eind, vwgt_np, vsize_np, ncommon, nparts)

