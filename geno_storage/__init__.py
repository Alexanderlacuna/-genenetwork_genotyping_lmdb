"""Genotype storage with cryptographic versioning."""

from .hashing import (
    canonical_json,
    compute_matrix_hash,
    compute_delta_hash,
    compute_metadata_hash,
    verify_hash_chain,
    hash_numpy_array
)

try:
    from .models import GenotypeMatrix, MatrixVersion
except ImportError:
    GenotypeMatrix = None
    MatrixVersion = None

try:
    from .delta import DeltaEncoder
except ImportError:
    DeltaEncoder = None

try:
    from .geno_parser import GenoParser
except ImportError:
    GenoParser = None

try:
    from .matrix_store import MatrixStore
except ImportError:
    MatrixStore = None

__all__ = [
    'MatrixStore',
    'MatrixVersion',
    'GenoParser',
    'GenotypeMatrix',
    'DeltaEncoder',
    'canonical_json',
    'compute_matrix_hash',
    'compute_delta_hash',
    'compute_metadata_hash',
    'verify_hash_chain',
    'hash_numpy_array',
]
