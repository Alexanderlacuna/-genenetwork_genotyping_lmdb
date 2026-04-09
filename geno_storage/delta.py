"""Delta encoding for matrix versioning."""

import struct
import numpy as np
from typing import Tuple, Union


class DeltaEncoder:
    """Encode/decode matrix deltas."""
    
    TYPE_FULL = 0x01
    TYPE_SPARSE = 0x02
    TYPE_XOR = 0x03
    
    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold
    
    def encode_full(self, matrix: np.ndarray) -> bytes:
        """Encode full matrix snapshot."""
        header = struct.pack('>BII', self.TYPE_FULL, matrix.shape[0], matrix.shape[1])
        dtype_byte = self._dtype_to_byte(matrix.dtype)
        return header + bytes([dtype_byte]) + matrix.tobytes()
    
    def encode_delta(self, base_matrix: np.ndarray, new_matrix: np.ndarray, version: int) -> bytes:
        """Encode delta, auto-selecting sparse or XOR."""
        if base_matrix.shape != new_matrix.shape:
            raise ValueError(f"Shape mismatch: {base_matrix.shape} vs {new_matrix.shape}")
        
        diff_mask = base_matrix != new_matrix
        change_ratio = np.sum(diff_mask) / base_matrix.size
        
        if change_ratio > self.threshold:
            return self._encode_xor(base_matrix, new_matrix, version)
        else:
            return self._encode_sparse(base_matrix, new_matrix, version)
    
    def _encode_sparse(self, base_matrix: np.ndarray, new_matrix: np.ndarray, version: int) -> bytes:
        """Sparse delta: store changed cells only."""
        diff_mask = base_matrix != new_matrix
        changed_rows = np.where(np.any(diff_mask, axis=1))[0]
        
        header = struct.pack(
            '>BIIII',
            self.TYPE_SPARSE,
            new_matrix.shape[0],
            new_matrix.shape[1],
            version,
            len(changed_rows)
        )
        
        dtype_byte = self._dtype_to_byte(new_matrix.dtype)
        data = header + bytes([dtype_byte])
        
        for row_idx in changed_rows:
            row_mask = diff_mask[row_idx]
            changed_cols = np.where(row_mask)[0]
            values = new_matrix[row_idx, changed_cols]
            
            data += struct.pack('>II', int(row_idx), len(changed_cols))
            data += changed_cols.astype(np.uint32).tobytes()
            data += values.tobytes()
        
        return data
    
    def _encode_xor(self, base_matrix: np.ndarray, new_matrix: np.ndarray, version: int) -> bytes:
        """XOR delta: store XOR mask for dense changes."""
        xor_mask = base_matrix ^ new_matrix
        header = struct.pack('>BIII', self.TYPE_XOR, new_matrix.shape[0], new_matrix.shape[1], version)
        dtype_byte = self._dtype_to_byte(new_matrix.dtype)
        return header + bytes([dtype_byte]) + xor_mask.tobytes()
    
    def decode(self, delta_bytes: bytes) -> Tuple[int, np.ndarray]:
        """Decode delta bytes."""
        delta_type = delta_bytes[0]
        
        if delta_type == self.TYPE_FULL:
            return self._decode_full(delta_bytes)
        elif delta_type == self.TYPE_SPARSE:
            return self._decode_sparse(delta_bytes)
        elif delta_type == self.TYPE_XOR:
            return self._decode_xor(delta_bytes)
        else:
            raise ValueError(f"Unknown delta type: {delta_type}")
    
    def _decode_full(self, delta_bytes: bytes) -> Tuple[int, np.ndarray]:
        """Decode full snapshot."""
        nrows, ncols = struct.unpack('>II', delta_bytes[1:9])
        dtype = self._byte_to_dtype(delta_bytes[9])
        matrix = np.frombuffer(delta_bytes[10:], dtype=dtype).reshape((nrows, ncols))
        return 1, matrix.copy()
    
    def _decode_sparse(self, delta_bytes: bytes) -> Tuple[int, dict]:
        """Decode sparse delta to dict."""
        nrows, ncols, version = struct.unpack('>III', delta_bytes[1:13])
        num_changed_rows = struct.unpack('>I', delta_bytes[13:17])[0]
        dtype = self._byte_to_dtype(delta_bytes[17])
        
        sparse_data = {'shape': (nrows, ncols), 'dtype': dtype, 'rows': {}}
        offset = 18
        dtype_size = np.dtype(dtype).itemsize
        
        for _ in range(num_changed_rows):
            row_idx, num_changes = struct.unpack('>II', delta_bytes[offset:offset+8])
            offset += 8
            
            cols = np.frombuffer(delta_bytes[offset:offset + num_changes * 4], dtype=np.uint32)
            offset += num_changes * 4
            
            values = np.frombuffer(delta_bytes[offset:offset + num_changes * dtype_size], dtype=dtype)
            offset += num_changes * dtype_size
            
            sparse_data['rows'][row_idx] = (cols, values)
        
        return version, sparse_data
    
    def _decode_xor(self, delta_bytes: bytes) -> Tuple[int, np.ndarray]:
        """Decode XOR mask."""
        nrows, ncols, version = struct.unpack('>III', delta_bytes[1:13])
        dtype = self._byte_to_dtype(delta_bytes[13])
        xor_mask = np.frombuffer(delta_bytes[14:], dtype=dtype).reshape((nrows, ncols))
        return version, xor_mask.copy()
    
    def apply_delta(self, base_matrix: np.ndarray, delta: Union[np.ndarray, dict], delta_type: int) -> np.ndarray:
        """Apply delta to base matrix."""
        new_matrix = base_matrix.copy()
        
        if delta_type == self.TYPE_SPARSE:
            for row_idx, (cols, values) in delta['rows'].items():
                new_matrix[row_idx, cols] = values
        elif delta_type == self.TYPE_XOR:
            new_matrix = new_matrix ^ delta
        else:
            raise ValueError(f"Cannot apply delta of type {delta_type}")
        
        return new_matrix
    
    def _dtype_to_byte(self, dtype: np.dtype) -> int:
        """Map dtype to byte."""
        mapping = {
            np.dtype('uint8'): 0x01,
            np.dtype('int8'): 0x02,
            np.dtype('uint16'): 0x03,
            np.dtype('int16'): 0x04,
            np.dtype('uint32'): 0x05,
            np.dtype('int32'): 0x06,
            np.dtype('float32'): 0x07,
            np.dtype('float64'): 0x08,
        }
        if dtype not in mapping:
            raise ValueError(f"Unsupported dtype: {dtype}")
        return mapping[dtype]
    
    def _byte_to_dtype(self, byte: int) -> np.dtype:
        """Map byte to dtype."""
        mapping = {
            0x01: np.uint8,
            0x02: np.int8,
            0x03: np.uint16,
            0x04: np.int16,
            0x05: np.uint32,
            0x06: np.int32,
            0x07: np.float32,
            0x08: np.float64,
        }
        if byte not in mapping:
            raise ValueError(f"Unknown dtype byte: {byte}")
        return mapping[byte]
    
    @staticmethod
    def get_storage_type(delta_bytes: bytes) -> str:
        """Get storage type name."""
        type_names = {0x01: 'full', 0x02: 'sparse', 0x03: 'xor'}
        return type_names.get(delta_bytes[0], 'unknown')
