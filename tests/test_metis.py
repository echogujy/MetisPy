"""
Tests for MetisPy
"""

import pytest
import numpy as np
from metispy import partition_graph, node_nd, partition_mesh_nodal, partition_mesh_dual


class TestBasicPartitioning:
    """Test basic graph partitioning functionality"""
    
    def test_simple_graph(self):
        """Test partitioning a simple 3-vertex graph"""
        xadj = np.array([0, 2, 4, 6], dtype=np.int64)
        adjncy = np.array([1, 2, 0, 2, 0, 1], dtype=np.int64)
        
        partition = partition_graph(xadj, adjncy, nparts=2)
        
        assert partition.shape == (3,)
        assert np.all(partition >= 0) and np.all(partition < 2)
    
    def test_kway_method(self):
        """Test K-way partitioning method"""
        xadj = np.array([0, 2, 4, 6], dtype=np.int64)
        adjncy = np.array([1, 2, 0, 2, 0, 1], dtype=np.int64)
        
        partition = partition_graph(xadj, adjncy, nparts=2, method='kway')
        
        assert partition.shape == (3,)
        assert np.all(partition >= 0) and np.all(partition < 2)
    
    def test_recursive_method(self):
        """Test recursive partitioning method"""
        xadj = np.array([0, 2, 4, 6], dtype=np.int64)
        adjncy = np.array([1, 2, 0, 2, 0, 1], dtype=np.int64)
        
        partition = partition_graph(xadj, adjncy, nparts=2, method='recursive')
        
        assert partition.shape == (3,)
        assert np.all(partition >= 0) and np.all(partition < 2)


class TestWeightedPartitioning:
    """Test partitioning with weights"""
    
    def test_vertex_weights(self):
        """Test partitioning with vertex weights"""
        xadj = np.array([0, 2, 4, 6], dtype=np.int64)
        adjncy = np.array([1, 2, 0, 2, 0, 1], dtype=np.int64)
        vwgt = np.array([1, 2, 1], dtype=np.int64)
        
        partition = partition_graph(xadj, adjncy, nparts=2, vwgt=vwgt)
        
        assert partition.shape == (3,)
        assert np.all(partition >= 0) and np.all(partition < 2)
    
    def test_edge_weights(self):
        """Test partitioning with edge weights"""
        xadj = np.array([0, 2, 4, 6], dtype=np.int64)
        adjncy = np.array([1, 2, 0, 2, 0, 1], dtype=np.int64)
        adjwgt = np.array([1, 1, 1, 1, 1, 1], dtype=np.int64)
        
        partition = partition_graph(xadj, adjncy, nparts=2, adjwgt=adjwgt)
        
        assert partition.shape == (3,)
        assert np.all(partition >= 0) and np.all(partition < 2)
    
    def test_both_weights(self):
        """Test partitioning with both vertex and edge weights"""
        xadj = np.array([0, 2, 4, 6], dtype=np.int64)
        adjncy = np.array([1, 2, 0, 2, 0, 1], dtype=np.int64)
        vwgt = np.array([1, 2, 1], dtype=np.int64)
        adjwgt = np.array([1, 1, 1, 1, 1, 1], dtype=np.int64)
        
        partition = partition_graph(xadj, adjncy, nparts=2, vwgt=vwgt, adjwgt=adjwgt)
        
        assert partition.shape == (3,)
        assert np.all(partition >= 0) and np.all(partition < 2)


class TestInputValidation:
    """Test input validation"""
    
    def test_invalid_dtype(self):
        """Test that invalid dtype raises error"""
        xadj = np.array([0, 2, 4, 6], dtype=np.int32)  # Wrong dtype
        adjncy = np.array([1, 2, 0, 2, 0, 1], dtype=np.int64)
        
        with pytest.raises(ValueError):
            partition_graph(xadj, adjncy, nparts=2)
    
    def test_invalid_method(self):
        """Test that invalid method raises error"""
        xadj = np.array([0, 2, 4, 6], dtype=np.int64)
        adjncy = np.array([1, 2, 0, 2, 0, 1], dtype=np.int64)
        
        with pytest.raises(ValueError):
            partition_graph(xadj, adjncy, nparts=2, method='invalid')
    
    def test_invalid_nparts(self):
        """Test that invalid nparts raises error"""
        xadj = np.array([0, 2, 4, 6], dtype=np.int64)
        adjncy = np.array([1, 2, 0, 2, 0, 1], dtype=np.int64)
        
        with pytest.raises(ValueError):
            partition_graph(xadj, adjncy, nparts=0)
    
    def test_non_array_input(self):
        """Test that non-array input raises error"""
        with pytest.raises(TypeError):
            partition_graph([0, 2, 4, 6], [1, 2, 0, 2, 0, 1], nparts=2)


class TestLargerGraphs:
    """Test with larger graphs"""
    
    def test_larger_graph(self):
        """Test partitioning a larger graph"""
        # Create a graph with 10 vertices
        xadj = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20], dtype=np.int64)
        adjncy = np.array([1, 2, 0, 2, 0, 1, 3, 4, 2, 5, 4, 6, 5, 7, 6, 8, 7, 9, 8, 9], dtype=np.int64)
        
        partition = partition_graph(xadj, adjncy, nparts=3)
        
        assert partition.shape == (10,)
        assert np.all(partition >= 0) and np.all(partition < 3)
    
    def test_multiple_partitions(self):
        """Test partitioning into multiple parts"""
        xadj = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16], dtype=np.int64)
        adjncy = np.array([1, 2, 0, 2, 0, 1, 3, 4, 2, 5, 4, 6, 5, 7, 6, 7], dtype=np.int64)
        
        partition = partition_graph(xadj, adjncy, nparts=4)
        
        assert partition.shape == (8,)
        assert np.all(partition >= 0) and np.all(partition < 4)


class TestNodeND:
    """Test sparse matrix/graph fill-reducing reordering"""
    
    def test_node_nd_basic(self):
        """Test basic matrix reordering"""
        xadj = np.array([0, 2, 4, 6], dtype=np.int64)
        adjncy = np.array([1, 2, 0, 2, 0, 1], dtype=np.int64)
        
        perm, iperm = node_nd(xadj, adjncy)
        
        assert perm.shape == (3,)
        assert iperm.shape == (3,)
        
        # Check that they are valid permutations
        assert sorted(perm.tolist()) == [0, 1, 2]
        assert sorted(iperm.tolist()) == [0, 1, 2]
        
        # Check inverse relationship
        for i in range(3):
            assert perm[iperm[i]] == i
            assert iperm[perm[i]] == i
            
    def test_node_nd_with_weights(self):
        """Test matrix reordering with vertex weights"""
        xadj = np.array([0, 2, 4, 6], dtype=np.int64)
        adjncy = np.array([1, 2, 0, 2, 0, 1], dtype=np.int64)
        vwgt = np.array([1, 2, 1], dtype=np.int64)
        
        perm, iperm = node_nd(xadj, adjncy, vwgt=vwgt)
        
        assert perm.shape == (3,)
        assert iperm.shape == (3,)
        assert sorted(perm.tolist()) == [0, 1, 2]
        
    def test_node_nd_validation(self):
        """Test input validations for node_nd"""
        xadj = np.array([0, 2, 4, 6], dtype=np.int64)
        adjncy = np.array([1, 2, 0, 2, 0, 1], dtype=np.int64)
        
        # Wrong dtype
        with pytest.raises(ValueError):
            node_nd(xadj.astype(np.int32), adjncy)
            
        # Wrong dimensions
        with pytest.raises(ValueError):
            node_nd(xadj.reshape(2, 2), adjncy)
            
        # Wrong weights size
        with pytest.raises(ValueError):
            node_nd(xadj, adjncy, vwgt=np.array([1, 2], dtype=np.int64))


class TestMeshPartitioning:
    """Test mesh partitioning functionality"""
    
    @pytest.fixture
    def simple_mesh(self):
        # Two quad elements sharing an edge (nodes 1 and 2)
        # Element 0: 0, 1, 2, 3
        # Element 1: 1, 4, 5, 2
        ne = 2
        nn = 6
        eptr = np.array([0, 4, 8], dtype=np.int64)
        eind = np.array([0, 1, 2, 3, 1, 4, 5, 2], dtype=np.int64)
        return ne, nn, eptr, eind
        
    def test_mesh_partitioning_nodal(self, simple_mesh):
        """Test nodal mesh partitioning"""
        ne, nn, eptr, eind = simple_mesh
        
        objval, epart, npart = partition_mesh_nodal(ne, nn, eptr, eind, nparts=2)
        
        assert isinstance(objval, int)
        assert epart.shape == (ne,)
        assert npart.shape == (nn,)
        
        assert np.all(epart >= 0) and np.all(epart < 2)
        assert np.all(npart >= 0) and np.all(npart < 2)
        
    def test_mesh_partitioning_dual(self, simple_mesh):
        """Test dual mesh partitioning"""
        ne, nn, eptr, eind = simple_mesh
        
        objval, epart, npart = partition_mesh_dual(ne, nn, eptr, eind, ncommon=2, nparts=2)
        
        assert isinstance(objval, int)
        assert epart.shape == (ne,)
        assert npart.shape == (nn,)
        
        assert np.all(epart >= 0) and np.all(epart < 2)
        assert np.all(npart >= 0) and np.all(npart < 2)
        
    def test_mesh_partitioning_nodal_with_weights(self, simple_mesh):
        """Test nodal mesh partitioning with node weights and sizes"""
        ne, nn, eptr, eind = simple_mesh
        vwgt = np.array([1, 2, 1, 1, 2, 1], dtype=np.int64)
        vsize = np.array([1, 1, 1, 1, 1, 1], dtype=np.int64)
        
        objval, epart, npart = partition_mesh_nodal(ne, nn, eptr, eind, vwgt=vwgt, vsize=vsize, nparts=2)
        
        assert isinstance(objval, int)
        assert epart.shape == (ne,)
        assert npart.shape == (nn,)
        
    def test_mesh_partitioning_dual_with_weights(self, simple_mesh):
        """Test dual mesh partitioning with element weights and sizes"""
        ne, nn, eptr, eind = simple_mesh
        vwgt = np.array([1, 2], dtype=np.int64)
        vsize = np.array([1, 1], dtype=np.int64)
        
        objval, epart, npart = partition_mesh_dual(ne, nn, eptr, eind, vwgt=vwgt, vsize=vsize, ncommon=2, nparts=2)
        
        assert isinstance(objval, int)
        assert epart.shape == (ne,)
        assert npart.shape == (nn,)
        
    def test_mesh_validation(self, simple_mesh):
        """Test input validation for mesh partitioning"""
        ne, nn, eptr, eind = simple_mesh
        
        # Invalid ne
        with pytest.raises(ValueError):
            partition_mesh_nodal(-1, nn, eptr, eind)
            
        # Invalid ncommon
        with pytest.raises(ValueError):
            partition_mesh_dual(ne, nn, eptr, eind, ncommon=0)
            
        # Wrong eptr length
        with pytest.raises(ValueError):
            partition_mesh_nodal(ne, nn, np.array([0, 4], dtype=np.int64), eind)
            
        # Mismatched vwgt shape (nodal expects nn elements)
        with pytest.raises(ValueError):
            partition_mesh_nodal(ne, nn, eptr, eind, vwgt=np.array([1, 2], dtype=np.int64))
            
        # Mismatched vwgt shape (dual expects ne elements)
        with pytest.raises(ValueError):
            partition_mesh_dual(ne, nn, eptr, eind, vwgt=np.array([1, 2, 3], dtype=np.int64))


if __name__ == "__main__":
    pytest.main()

