"""Core data models for genotype storage.

These dataclasses represent the immutable facts at the center of the
architecture. All other modules depend on these types.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np


@dataclass
class GenotypeMatrix:
    """Genotype matrix with metadata — immutable data."""
    matrix: np.ndarray
    markers: List[str]
    samples: List[str]
    chromosomes: List[str]
    cM: List[Optional[float]]
    Mb: List[Optional[float]]
    allele_map: Dict[str, int]
    founders: List[str]
    het_code: int
    unk_code: int
    dataset_name: str = ""
    cross_type: str = ""
    mat_allele: Optional[str] = None
    pat_allele: Optional[str] = None
    genome_build: str = ""
    file_metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        # Freeze the numpy array to prevent accidental mutation.
        # Copy-on-write: callers must explicitly copy if they need to modify.
        if self.matrix.flags.writeable:
            self.matrix = self.matrix.copy()
            self.matrix.flags.writeable = False

    def get_allele_frequencies(self) -> Dict[int, int]:
        """Count allele codes."""
        unique, counts = np.unique(self.matrix, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))

    def decode_row(self, row_idx: int) -> List[str]:
        """Decode numeric codes to symbols."""
        reverse_map = {v: k for k, v in self.allele_map.items()}
        reverse_map[self.het_code] = 'H'
        reverse_map[self.unk_code] = 'U'
        return [reverse_map.get(code, '?') for code in self.matrix[row_idx]]


@dataclass
class MatrixVersion:
    """Matrix version entry — immutable fact about an event."""
    dataset_id: str
    matrix_version: int
    matrix_hash: str
    prev_matrix_hash: Optional[str]
    storage_type: str
    payload: bytes
    timestamp: str
    reason: str
    author: str
    nrows: int
    ncols: int
    dtype: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict (excludes payload)."""
        return {
            'dataset_id': self.dataset_id,
            'matrix_version': self.matrix_version,
            'matrix_hash': self.matrix_hash,
            'prev_matrix_hash': self.prev_matrix_hash,
            'storage_type': self.storage_type,
            'timestamp': self.timestamp,
            'reason': self.reason,
            'author': self.author,
            'nrows': self.nrows,
            'ncols': self.ncols,
            'dtype': self.dtype,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], payload: bytes) -> 'MatrixVersion':
        """Create from dict and payload."""
        return cls(
            dataset_id=data['dataset_id'],
            matrix_version=data['matrix_version'],
            matrix_hash=data['matrix_hash'],
            prev_matrix_hash=data.get('prev_matrix_hash'),
            storage_type=data['storage_type'],
            payload=payload,
            timestamp=data['timestamp'],
            reason=data['reason'],
            author=data['author'],
            nrows=data['nrows'],
            ncols=data['ncols'],
            dtype=data['dtype'],
        )
