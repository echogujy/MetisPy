"""
Tests for MetisPy
"""

import pytest
import numpy as np
from metispy import partition_graph


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


if __name__ == "__main__":
    pytest.main()
