"""Cryptographic hashing utilities."""

import hashlib
import json
from typing import Dict, Any, Optional
import numpy as np


def canonical_json(data: Dict[str, Any]) -> bytes:
    """Deterministic JSON for hashing."""
    return json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')


def compute_matrix_hash(payload_binary: bytes, prev_matrix_hash: Optional[str] = None, version: int = 1) -> str:
    """SHA256("MATRIX_V1" || payload || prev_hash)."""
    hasher = hashlib.sha256()
    hasher.update(f"MATRIX_V{version}".encode('utf-8'))
    hasher.update(payload_binary)
    if prev_matrix_hash:
        hasher.update(prev_matrix_hash.encode('utf-8'))
    return hasher.hexdigest()


def compute_delta_hash(delta_binary: bytes, base_matrix_hash: str, version: int = 1) -> str:
    """SHA256("DELTA_V1" || delta || base_hash)."""
    hasher = hashlib.sha256()
    hasher.update(f"DELTA_V{version}".encode('utf-8'))
    hasher.update(delta_binary)
    hasher.update(base_matrix_hash.encode('utf-8'))
    return hasher.hexdigest()


def compute_metadata_hash(tool: str, matrix_hash: str, semantic_fields: Dict[str, Any], version: int = 1) -> str:
    """SHA256("META_V1" || tool || matrix_hash || fields)."""
    hasher = hashlib.sha256()
    hasher.update(f"META_V{version}".encode('utf-8'))
    hasher.update(tool.encode('utf-8'))
    hasher.update(matrix_hash.encode('utf-8'))
    hasher.update(canonical_json(semantic_fields))
    return hasher.hexdigest()


def verify_hash_chain(entries: list, hash_field: str = 'matrix_hash', prev_hash_field: str = 'prev_matrix_hash') -> tuple[bool, Optional[int]]:
    """Verify hash chain integrity."""
    for i, entry in enumerate(entries):
        current_hash = entry.get(hash_field)
        prev_hash = entry.get(prev_hash_field)
        
        if i == 0:
            if prev_hash is not None:
                return False, i
        else:
            expected_prev = entries[i-1].get(hash_field)
            if prev_hash != expected_prev:
                return False, i
                
    return True, None


def hash_numpy_array(arr: np.ndarray) -> str:
    """Quick integrity hash for numpy arrays."""
    hasher = hashlib.sha256()
    hasher.update(arr.tobytes())
    hasher.update(str(arr.shape).encode('utf-8'))
    hasher.update(str(arr.dtype).encode('utf-8'))
    return hasher.hexdigest()
