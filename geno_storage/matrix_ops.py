"""Matrix operations — pure functions operating on GenotypeMatrix data.

In FP style, data is inert. These functions take data as input and
produce new data as output. No mutation, no side effects.
"""

from typing import Dict, List

import numpy as np

from .models import GenotypeMatrix


def get_allele_frequencies(genotype: GenotypeMatrix) -> Dict[int, int]:
    """Count allele codes."""
    unique, counts = np.unique(genotype.matrix, return_counts=True)
    return dict(zip(unique.tolist(), counts.tolist()))


def decode_row(genotype: GenotypeMatrix, row_idx: int) -> List[str]:
    """Decode numeric codes to symbols."""
    reverse_map = {v: k for k, v in genotype.allele_map.items()}
    reverse_map[genotype.het_code] = 'H'
    reverse_map[genotype.unk_code] = 'U'
    return [reverse_map.get(code, '?') for code in genotype.matrix[row_idx]]
