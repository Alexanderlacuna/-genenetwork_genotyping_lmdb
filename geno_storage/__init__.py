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
    from .delta import DeltaEncoder
except ImportError:
    DeltaEncoder = None

try:
    from .geno_parser import GenoParser, GenotypeMatrix
except ImportError:
    GenoParser = None
    GenotypeMatrix = None

try:
    from .matrix_store import MatrixStore, MatrixVersion
except ImportError:
    MatrixStore = None
    MatrixVersion = None

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