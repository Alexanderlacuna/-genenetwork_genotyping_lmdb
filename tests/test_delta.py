"""
Tests for delta encoding/decoding.
"""

import pytest
import numpy as np
from geno_storage.delta import DeltaEncoder


class TestFullSnapshot:
    """Test full snapshot encoding/decoding."""
    
    def test_encode_decode_uint8(self):
        """Full snapshot should round-trip correctly."""
        encoder = DeltaEncoder()
        matrix = np.array([[0, 1, 2], [3, 2, 1]], dtype=np.uint8)
        
        encoded = encoder.encode_full(matrix)
        version, decoded = encoder.decode(encoded)
        
        assert version == 1
        np.testing.assert_array_equal(decoded, matrix)
    
    def test_encode_decode_float(self):
        """Full snapshot should work with float arrays."""
        encoder = DeltaEncoder()
        matrix = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        
        encoded = encoder.encode_full(matrix)
        version, decoded = encoder.decode(encoded)
        
        np.testing.assert_array_almost_equal(decoded, matrix)
    
    def test_large_matrix(self):
        """Full snapshot should handle large matrices."""
        encoder = DeltaEncoder()
        matrix = np.random.randint(0, 4, size=(1000, 100), dtype=np.uint8)
        
        encoded = encoder.encode_full(matrix)
        version, decoded = encoder.decode(encoded)
        
        np.testing.assert_array_equal(decoded, matrix)


class TestSparseDelta:
    """Test sparse delta encoding."""
    
    def test_sparse_changes(self):
        """Sparse changes should use sparse encoding."""
        encoder = DeltaEncoder(threshold=0.5)  # High threshold to force sparse
        
        base = np.array([[0, 0, 0], [0, 0, 0]], dtype=np.uint8)
        new = np.array([[1, 0, 0], [0, 2, 0]], dtype=np.uint8)  # 2 changes
        
        encoded = encoder.encode_delta(base, new, version=2)
        storage_type = DeltaEncoder.get_storage_type(encoded)
        
        assert storage_type == 'sparse'
    
    def test_sparse_roundtrip(self):
        """Sparse delta should apply correctly."""
        encoder = DeltaEncoder(threshold=0.5)
        
        base = np.zeros((10, 10), dtype=np.uint8)
        base[0, 0] = 1
        base[5, 5] = 2
        
        new = base.copy()
        new[3, 3] = 3
        new[7, 7] = 1
        
        encoded = encoder.encode_delta(base, new, version=2)
        version, delta_data = encoder.decode(encoded)
        
        # Apply delta
        result = encoder.apply_delta(base, delta_data, encoded[0])
        
        np.testing.assert_array_equal(result, new)
    
    def test_single_row_changes(self):
        """Changes in single row should work."""
        encoder = DeltaEncoder(threshold=0.5)
        
        base = np.zeros((5, 5), dtype=np.uint8)
        new = base.copy()
        new[2, :] = [1, 2, 3, 4, 5]
        
        encoded = encoder.encode_delta(base, new, version=2)
        version, delta_data = encoder.decode(encoded)
        result = encoder.apply_delta(base, delta_data, encoded[0])
        
        np.testing.assert_array_equal(result, new)


class TestXORDelta:
    """Test XOR delta encoding."""
    
    def test_dense_changes(self):
        """Dense changes should use XOR encoding."""
        encoder = DeltaEncoder(threshold=0.1)  # Low threshold to force XOR
        
        base = np.zeros((10, 10), dtype=np.uint8)
        new = np.random.randint(0, 4, size=(10, 10), dtype=np.uint8)
        
        encoded = encoder.encode_delta(base, new, version=2)
        storage_type = DeltaEncoder.get_storage_type(encoded)
        
        assert storage_type == 'xor'
    
    def test_xor_roundtrip(self):
        """XOR delta should apply correctly."""
        encoder = DeltaEncoder(threshold=0.1)
        
        base = np.ones((3, 3), dtype=np.uint8)
        new = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]], dtype=np.uint8)
        
        encoded = encoder.encode_delta(base, new, version=2)
        version, xor_mask = encoder.decode(encoded)
        
        result = encoder.apply_delta(base, xor_mask, encoded[0])
        
        np.testing.assert_array_equal(result, new)


class TestDeltaApply:
    """Test delta application."""
    
    def test_shape_mismatch(self):
        """Should raise error on shape mismatch."""
        encoder = DeltaEncoder()
        
        base = np.zeros((5, 5), dtype=np.uint8)
        new = np.zeros((6, 6), dtype=np.uint8)
        
        with pytest.raises(ValueError):
            encoder.encode_delta(base, new, version=2)
    
    def test_multiple_deltas(self):
        """Should apply multiple deltas sequentially."""
        encoder = DeltaEncoder()
        
        # Start with base
        base = np.zeros((5, 5), dtype=np.uint8)
        
        # Apply delta 1
        v1 = base.copy()
        v1[0, 0] = 1
        enc1 = encoder.encode_delta(base, v1, version=2)
        _, delta1 = encoder.decode(enc1)
        result1 = encoder.apply_delta(base, delta1, enc1[0])
        
        # Apply delta 2 to result1
        v2 = result1.copy()
        v2[1, 1] = 2
        enc2 = encoder.encode_delta(result1, v2, version=3)
        _, delta2 = encoder.decode(enc2)
        result2 = encoder.apply_delta(result1, delta2, enc2[0])
        
        assert result2[0, 0] == 1
        assert result2[1, 1] == 2


class TestStorageType:
    """Test storage type detection."""
    
    def test_full_type(self):
        """Should detect full snapshot type."""
        encoder = DeltaEncoder()
        matrix = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        
        encoded = encoder.encode_full(matrix)
        
        assert DeltaEncoder.get_storage_type(encoded) == 'full'
    
    def test_sparse_type(self):
        """Should detect sparse type."""
        encoder = DeltaEncoder(threshold=0.5)
        base = np.zeros((10, 10), dtype=np.uint8)
        new = base.copy()
        new[0, 0] = 1
        
        encoded = encoder.encode_delta(base, new, version=2)
        
        assert DeltaEncoder.get_storage_type(encoded) == 'sparse'
    
    def test_unknown_type(self):
        """Should return unknown for invalid data."""
        invalid = b'\xff\x00\x00\x00'  # Invalid type byte
        
        assert DeltaEncoder.get_storage_type(invalid) == 'unknown'


class TestEdgeCases:
    """Test edge cases."""
    
    def test_empty_matrix(self):
        """Should handle empty-ish matrices."""
        encoder = DeltaEncoder()
        
        base = np.zeros((1, 1), dtype=np.uint8)
        new = np.array([[1]], dtype=np.uint8)
        
        encoded = encoder.encode_delta(base, new, version=2)
        version, delta = encoder.decode(encoded)
        result = encoder.apply_delta(base, delta, encoded[0])
        
        np.testing.assert_array_equal(result, new)
    
    def test_no_changes(self):
        """Should handle no changes."""
        encoder = DeltaEncoder(threshold=0.5)
        
        base = np.ones((5, 5), dtype=np.uint8)
        new = base.copy()
        
        # With no changes, it will still encode but as sparse (0 changes)
        encoded = encoder.encode_delta(base, new, version=2)
        version, delta = encoder.decode(encoded)
        result = encoder.apply_delta(base, delta, encoded[0])
        
        np.testing.assert_array_equal(result, new)
