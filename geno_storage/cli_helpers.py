"""CLI helper utilities.

Higher-order patterns for common CLI workflows.
"""

from contextlib import contextmanager
from typing import Optional, Tuple

from .matrix_store import MatrixStore
from .models import GenotypeMatrix


@contextmanager
def resolved_matrix(
    lmdb_path: str,
    dataset_id: str,
    version: Optional[int] = None,
    read_only: bool = True
):
    """Context manager: open store, resolve version, yield matrix.

    Eliminates duplicated boilerplate in CLI commands.

    Yields:
        Tuple of (MatrixStore, resolved_version, GenotypeMatrix)
    """
    with MatrixStore(lmdb_path, read_only=read_only) as store:
        target_ver = version or store.get_current_version(dataset_id)[0]
        matrix = store.get_matrix(dataset_id, target_ver)
        yield store, target_ver, matrix
