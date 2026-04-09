"""
Tests for cryptographic hashing functions.
"""

import pytest
import numpy as np
from geno_storage.hashing import (
    canonical_json,
    compute_matrix_hash,
    compute_delta_hash,
    compute_metadata_hash,
    verify_hash_chain,
    hash_numpy_array
)


class TestCanonicalJson:
    """Test canonical JSON serialization."""
    
    def test_simple_dict(self):
        """Canonical JSON should sort keys."""
        data = {"z": 1, "a": 2, "m": 3}
        result = canonical_json(data)
        assert result == b'{"a":2,"m":3,"z":1}'
    
    def test_nested_dict(self):
        """Canonical JSON should handle nested structures."""
        data = {"outer": {"z": 1, "a": 2}}
        result = canonical_json(data)
        assert result == b'{"outer":{"a":2,"z":1}}'
    
    def test_determinism(self):
        """Same data should always produce same output."""
        data1 = {"b": 1, "a": 2}
        data2 = {"a": 2, "b": 1}
        assert canonical_json(data1) == canonical_json(data2)
    
    def test_unicode(self):
        """Should handle Unicode correctly."""
        data = {"name": "test_value"}
        result = canonical_json(data)
        assert result == b'{"name":"test_value"}'


class TestMatrixHash:
    """Test matrix hash computation."""
    
    def test_v1_no_prev(self):
        """v1 matrix hash with no previous hash."""
        payload = b"test_matrix_data"
        hash1 = compute_matrix_hash(payload, None, version=1)
        
        # Should be deterministic
        hash2 = compute_matrix_hash(payload, None, version=1)
        assert hash1 == hash2
        
        # Should be different with different version
        hash3 = compute_matrix_hash(payload, None, version=2)
        assert hash1 != hash3
    
    def test_chained_hash(self):
        """Hash chain should depend on previous hash."""
        payload = b"data"
        prev_hash = "abc123"
        
        hash1 = compute_matrix_hash(payload, prev_hash)
        hash2 = compute_matrix_hash(payload, "different_prev")
        
        assert hash1 != hash2
    
    def test_domain_separation(self):
        """Matrix hash should be domain-separated."""
        payload = b"data"
        
        matrix_hash = compute_matrix_hash(payload, None)
        delta_hash = compute_delta_hash(payload, "base")
        meta_hash = compute_metadata_hash("tool", "hash", {})
        
        # All should be different
        assert matrix_hash != delta_hash
        assert matrix_hash != meta_hash
        assert delta_hash != meta_hash


class TestDeltaHash:
    """Test delta hash computation."""
    
    def test_base_binding(self):
        """Delta hash should bind to base matrix."""
        payload = b"delta_data"
        base1 = "base_hash_1"
        base2 = "base_hash_2"
        
        hash1 = compute_delta_hash(payload, base1)
        hash2 = compute_delta_hash(payload, base2)
        
        assert hash1 != hash2


class TestMetadataHash:
    """Test metadata hash computation."""
    
    def test_semantic_binding(self):
        """Metadata hash should bind to matrix and tool."""
        fields = {"encoding": "AA/AB/BB", "cross_type": "f2"}
        
        hash1 = compute_metadata_hash("rqtl", "matrix_hash_1", fields)
        hash2 = compute_metadata_hash("gemma", "matrix_hash_1", fields)
        hash3 = compute_metadata_hash("rqtl", "matrix_hash_2", fields)
        
        # Different tool = different hash
        assert hash1 != hash2
        # Different matrix = different hash
        assert hash1 != hash3
    
    def test_field_variation(self):
        """Changing semantic fields should change hash."""
        hash1 = compute_metadata_hash("rqtl", "hash", {"field": "value1"})
        hash2 = compute_metadata_hash("rqtl", "hash", {"field": "value2"})
        
        assert hash1 != hash2


class TestHashChainVerification:
    """Test hash chain verification."""
    
    def test_valid_chain(self):
        """Valid chain should pass verification."""
        entries = [
            {"matrix_hash": "h1", "prev_matrix_hash": None},
            {"matrix_hash": "h2", "prev_matrix_hash": "h1"},
            {"matrix_hash": "h3", "prev_matrix_hash": "h2"},
        ]
        
        valid, idx = verify_hash_chain(entries)
        assert valid is True
        assert idx is None
    
    def test_broken_chain(self):
        """Broken chain should be detected."""
        entries = [
            {"matrix_hash": "h1", "prev_matrix_hash": None},
            {"matrix_hash": "h2", "prev_matrix_hash": "wrong_hash"},
            {"matrix_hash": "h3", "prev_matrix_hash": "h2"},
        ]
        
        valid, idx = verify_hash_chain(entries)
        assert valid is False
        assert idx == 1  # Broken at index 1
    
    def test_v1_with_prev(self):
        """v1 should not have prev_hash."""
        entries = [
            {"matrix_hash": "h1", "prev_matrix_hash": "something"},
        ]
        
        valid, idx = verify_hash_chain(entries)
        assert valid is False
        assert idx == 0
    
    def test_empty_chain(self):
        """Empty chain should be valid."""
        valid, idx = verify_hash_chain([])
        assert valid is True
        assert idx is None


class TestNumpyArrayHash:
    """Test numpy array hashing."""
    
    def test_same_array(self):
        """Same array should have same hash."""
        arr = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        hash1 = hash_numpy_array(arr)
        hash2 = hash_numpy_array(arr.copy())
        
        assert hash1 == hash2
    
    def test_different_values(self):
        """Different values should have different hash."""
        arr1 = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        arr2 = np.array([[1, 2], [3, 5]], dtype=np.uint8)
        
        hash1 = hash_numpy_array(arr1)
        hash2 = hash_numpy_array(arr2)
        
        assert hash1 != hash2
    
    def test_different_shape(self):
        """Different shape should have different hash."""
        arr1 = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        arr2 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
        
        hash1 = hash_numpy_array(arr1)
        hash2 = hash_numpy_array(arr2)
        
        assert hash1 != hash2
