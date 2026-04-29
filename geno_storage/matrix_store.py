"""LMDB storage for versioned genotype matrices.

Actions (I/O) are isolated in this module. Calculations are extracted
to module-level functions where possible.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import lmdb

from .hashing import compute_matrix_hash, canonical_json
from .delta import DeltaEncoder
from .models import GenotypeMatrix, MatrixVersion


def encode_metadata(genotype_matrix: GenotypeMatrix) -> Dict:
    """Encode GenotypeMatrix metadata to dict.

    Pure calculation: no side effects, no I/O.
    """
    return {
        'markers': genotype_matrix.markers,
        'samples': genotype_matrix.samples,
        'chromosomes': genotype_matrix.chromosomes,
        'cM': genotype_matrix.cM,
        'Mb': genotype_matrix.Mb,
        'allele_map': genotype_matrix.allele_map,
        'founders': genotype_matrix.founders,
        'het_code': genotype_matrix.het_code,
        'unk_code': genotype_matrix.unk_code,
        'dataset_name': genotype_matrix.dataset_name,
        'cross_type': genotype_matrix.cross_type,
        'mat_allele': genotype_matrix.mat_allele,
        'pat_allele': genotype_matrix.pat_allele,
    }


def decode_metadata(data: Dict) -> Dict:
    """Decode metadata dict (handle missing fields for backwards compat).

    Pure calculation: no side effects, no I/O.
    """
    # Backwards compat: old stores used 'positions' for a single column
    cm = data.get('cM')
    mb = data.get('Mb')
    if cm is None and mb is None and 'positions' in data:
        old_positions = data.get('positions', [])
        cm = old_positions
        mb = old_positions

    return {
        'markers': data.get('markers', []),
        'samples': data.get('samples', []),
        'chromosomes': data.get('chromosomes', []),
        'cM': cm if cm is not None else [],
        'Mb': mb if mb is not None else [],
        'allele_map': data.get('allele_map', {}),
        'founders': data.get('founders', []),
        'het_code': data.get('het_code', 2),
        'unk_code': data.get('unk_code', 3),
        'dataset_name': data.get('dataset_name', ''),
        'cross_type': data.get('cross_type', ''),
        'mat_allele': data.get('mat_allele'),
        'pat_allele': data.get('pat_allele'),
    }


class MatrixStore:
    """LMDB storage for versioned matrices.

    Actions are methods on this class. Calculations are either
    module-level functions or methods that delegate to them.
    """

    DB_MATRIX_HISTORY = b'matrix_history'
    DB_GENOTYPES = b'genotypes'
    DB_INFO = b'info'
    FULL_SNAPSHOT_INTERVAL = 10

    def __init__(self, db_path: Union[str, Path], map_size: int = 100 * 1024 * 1024 * 1024, read_only: bool = False):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)

        open_kwargs = {'map_size': map_size, 'max_dbs': 10}
        if read_only:
            open_kwargs['readonly'] = True

        self.env = lmdb.open(str(self.db_path), **open_kwargs)
        self.delta_encoder = DeltaEncoder()
        self.read_only = read_only

        with self.env.begin(write=not read_only) as txn:
            self.env.open_db(self.DB_MATRIX_HISTORY, txn=txn, create=not read_only)
            self.env.open_db(self.DB_GENOTYPES, txn=txn, create=not read_only)
            self.env.open_db(self.DB_INFO, txn=txn, create=not read_only)

    def get_db(self, txn, db_name: bytes):
        """Get DB handle for transaction."""
        return self.env.open_db(db_name, txn=txn, create=not self.read_only)

    def make_history_key(self, dataset_id: str, version: int) -> bytes:
        """Make LMDB key for a version entry."""
        return f"{dataset_id}:{version:010d}".encode('utf-8')

    def make_info_key(self, dataset_id: str) -> bytes:
        """Make LMDB key for info entry."""
        return dataset_id.encode('utf-8')

    def should_store_full_snapshot(self, version: int) -> bool:
        """Check if this version should be a full snapshot."""
        if version == 1:
            return True
        return (version - 1) % self.FULL_SNAPSHOT_INTERVAL == 0

    def store_initial(
        self,
        dataset_id: str,
        genotype_matrix: GenotypeMatrix,
        author: str = "import",
        reason: str = "Initial import",
        timestamp: Optional[str] = None
    ) -> MatrixVersion:
        """Store initial matrix (version 1).

        Args:
            timestamp: ISO format timestamp. Defaults to current UTC time.
                      Pass an explicit value for deterministic testing.
        """
        payload = self.delta_encoder.encode_full(genotype_matrix.matrix)
        matrix_hash = compute_matrix_hash(payload, None)

        ts = timestamp or datetime.utcnow().isoformat() + 'Z'

        version = MatrixVersion(
            dataset_id=dataset_id,
            matrix_version=1,
            matrix_hash=matrix_hash,
            prev_matrix_hash=None,
            storage_type='full',
            payload=payload,
            timestamp=ts,
            reason=reason,
            author=author,
            nrows=genotype_matrix.matrix.shape[0],
            ncols=genotype_matrix.matrix.shape[1],
            dtype=str(genotype_matrix.matrix.dtype)
        )

        with self.env.begin(write=True) as txn:
            matrix_db = self.get_db(txn, self.DB_MATRIX_HISTORY)
            geno_db = self.get_db(txn, self.DB_GENOTYPES)
            info_db = self.get_db(txn, self.DB_INFO)

            key = self.make_history_key(dataset_id, 1)
            txn.put(key, canonical_json(version.to_dict()), db=matrix_db)
            txn.put(key + b':payload', payload, db=matrix_db)
            txn.put(key + b':metadata', canonical_json(encode_metadata(genotype_matrix)), db=matrix_db)

            txn.put(self.make_info_key(dataset_id), canonical_json({
                'current_version': 1,
                'current_hash': matrix_hash,
                'nrows': version.nrows,
                'ncols': version.ncols,
                'dtype': version.dtype
            }), db=geno_db)

            self.update_info_head(txn, dataset_id, version, info_db)

        return version

    def store_update(
        self,
        dataset_id: str,
        new_matrix: Union[np.ndarray, GenotypeMatrix],
        author: str,
        reason: str,
        timestamp: Optional[str] = None
    ) -> MatrixVersion:
        """Store updated matrix as new version.

        Args:
            timestamp: ISO format timestamp. Defaults to current UTC time.
                      Pass an explicit value for deterministic testing.
        """
        current_version, current_hash = self.get_current_version(dataset_id)
        new_version_num = current_version + 1

        # Extract numpy array if GenotypeMatrix passed
        if isinstance(new_matrix, GenotypeMatrix):
            new_matrix_data = new_matrix.matrix
        else:
            new_matrix_data = new_matrix

        if self.should_store_full_snapshot(new_version_num):
            payload = self.delta_encoder.encode_full(new_matrix_data)
            storage_type = 'full'
        else:
            try:
                base_genotype = self.get_matrix(dataset_id, current_version)
                base_matrix_data = base_genotype.matrix
                payload = self.delta_encoder.encode_delta(base_matrix_data, new_matrix_data, new_version_num)
                storage_type = 'delta'
            except ValueError:
                payload = self.delta_encoder.encode_full(new_matrix_data)
                storage_type = 'full'

        matrix_hash = compute_matrix_hash(payload, current_hash)

        ts = timestamp or datetime.utcnow().isoformat() + 'Z'

        version = MatrixVersion(
            dataset_id=dataset_id,
            matrix_version=new_version_num,
            matrix_hash=matrix_hash,
            prev_matrix_hash=current_hash,
            storage_type=storage_type,
            payload=payload,
            timestamp=ts,
            reason=reason,
            author=author,
            nrows=new_matrix_data.shape[0],
            ncols=new_matrix_data.shape[1],
            dtype=str(new_matrix_data.dtype)
        )

        with self.env.begin(write=True) as txn:
            matrix_db = self.get_db(txn, self.DB_MATRIX_HISTORY)
            geno_db = self.get_db(txn, self.DB_GENOTYPES)
            info_db = self.get_db(txn, self.DB_INFO)

            key = self.make_history_key(dataset_id, new_version_num)
            txn.put(key, canonical_json(version.to_dict()), db=matrix_db)
            txn.put(key + b':payload', payload, db=matrix_db)

            # Write metadata on full snapshots when GenotypeMatrix is provided
            if storage_type == 'full' and isinstance(new_matrix, GenotypeMatrix):
                txn.put(key + b':metadata', canonical_json(encode_metadata(new_matrix)), db=matrix_db)

            txn.put(self.make_info_key(dataset_id), canonical_json({
                'current_version': new_version_num,
                'current_hash': matrix_hash,
                'nrows': new_matrix_data.shape[0],
                'ncols': new_matrix_data.shape[1],
                'dtype': str(new_matrix_data.dtype)
            }), db=geno_db)

            self.update_info_head(txn, dataset_id, version, info_db)

        return version

    def update_info_head(self, txn, dataset_id: str, version: MatrixVersion, info_db) -> None:
        """Update HEAD pointer."""
        info_key = self.make_info_key(dataset_id)
        existing = txn.get(info_key, db=info_db)
        info = json.loads(existing.decode('utf-8')) if existing else {'dataset_id': dataset_id, 'description': '', 'author': ''}

        info['current_matrix_version'] = version.matrix_version
        info['current_matrix_hash'] = version.matrix_hash
        info['last_updated'] = version.timestamp

        txn.put(info_key, canonical_json(info), db=info_db)

    def get_version(self, dataset_id: str, version: int) -> Optional[MatrixVersion]:
        """Get version entry."""
        with self.env.begin() as txn:
            matrix_db = self.get_db(txn, self.DB_MATRIX_HISTORY)
            key = self.make_history_key(dataset_id, version)
            value = txn.get(key, db=matrix_db)
            if not value:
                return None

            payload = txn.get(key + b':payload', db=matrix_db)
            return MatrixVersion.from_dict(json.loads(value.decode('utf-8')), payload)

    def get_matrix(self, dataset_id: str, target_version: Optional[int] = None) -> GenotypeMatrix:
        """Reconstruct matrix with metadata at version."""
        if target_version is None:
            target_version, _ = self.get_current_version(dataset_id)

        # Phase 1: Plan — what versions do we need? (calculation)
        plan = self.reconstruction_plan(dataset_id, target_version)

        # Phase 2: Fetch — get payloads from LMDB (action)
        payloads = self.fetch_payloads(plan)

        # Phase 3: Reconstruct — assemble matrix (calculation)
        return self.reconstruct_from_payloads(plan, payloads)

    def reconstruction_plan(self, dataset_id: str, target_version: int) -> Dict:
        """Determine which versions are needed for reconstruction."""
        full_version = self.find_nearest_full_snapshot(dataset_id, target_version)
        if full_version is None:
            raise ValueError(
                f"Cannot reconstruct version {target_version} of dataset '{dataset_id}': "
                f"no full snapshot found.\n"
                f"  - This usually means the dataset was never imported, or all full snapshots were deleted.\n"
                f"  - The ledger may be corrupted (try 'verify' to check)."
            )

        needed_versions = [full_version] + list(range(full_version + 1, target_version + 1))
        return {
            'dataset_id': dataset_id,
            'target_version': target_version,
            'full_version': full_version,
            'needed_versions': needed_versions,
        }

    def fetch_payloads(self, plan: Dict) -> Dict[int, bytes]:
        """Fetch all required payloads from LMDB."""
        dataset_id = plan['dataset_id']
        payloads = {}

        for v in plan['needed_versions']:
            version_data = self.get_version(dataset_id, v)
            if not version_data:
                full_version = plan['full_version']
                target_version = plan['target_version']
                raise ValueError(
                    f"Cannot reconstruct version {target_version} of dataset '{dataset_id}': "
                    f"{'full snapshot' if v == full_version else 'delta'} at v{v} is referenced but missing.\n"
                    f"  - The nearest full snapshot is at v{full_version}.\n"
                    f"  - The ledger may be corrupted (try 'verify' to check)."
                )
            if version_data.payload is None:
                full_version = plan['full_version']
                target_version = plan['target_version']
                raise ValueError(
                    f"Cannot reconstruct version {target_version} of dataset '{dataset_id}': "
                    f"payload for {'full snapshot' if v == full_version else 'delta'} at v{v} is missing.\n"
                    f"  - The ledger may be corrupted (try 'verify' to check).\n"
                    f"  - If you have an external backup, restore it."
                )
            payloads[v] = version_data.payload

        return payloads

    def reconstruct_from_payloads(self, plan: Dict, payloads: Dict[int, bytes]) -> GenotypeMatrix:
        """Reconstruct GenotypeMatrix from payloads.

        Pure calculation: given payloads and plan, assemble matrix.
        No LMDB I/O, no side effects.
        """
        dataset_id = plan['dataset_id']
        full_version = plan['full_version']
        target_version = plan['target_version']

        # Get metadata from full snapshot
        metadata = self.get_metadata(dataset_id, full_version)

        # Decode full snapshot
        _, current_matrix = self.delta_encoder.decode(payloads[full_version])

        # Apply deltas
        for v in range(full_version + 1, target_version + 1):
            _, delta_data = self.delta_encoder.decode(payloads[v])
            delta_type = payloads[v][0]
            current_matrix = self.delta_encoder.apply_delta(current_matrix, delta_data, delta_type)

        # Return GenotypeMatrix with metadata
        return GenotypeMatrix(
            matrix=current_matrix,
            markers=metadata['markers'],
            samples=metadata['samples'],
            chromosomes=metadata['chromosomes'],
            cM=metadata['cM'],
            Mb=metadata['Mb'],
            allele_map=metadata['allele_map'],
            founders=metadata['founders'],
            het_code=metadata['het_code'],
            unk_code=metadata['unk_code'],
            dataset_name=metadata['dataset_name'],
            cross_type=metadata['cross_type'],
            mat_allele=metadata['mat_allele'],
            pat_allele=metadata['pat_allele'],
        )

    def get_metadata(self, dataset_id: str, version: int) -> Dict:
        """Get metadata for a version."""
        with self.env.begin() as txn:
            matrix_db = self.get_db(txn, self.DB_MATRIX_HISTORY)
            key = self.make_history_key(dataset_id, version) + b':metadata'
            value = txn.get(key, db=matrix_db)
            if value:
                return decode_metadata(json.loads(value.decode('utf-8')))
            # Fallback: return empty metadata for backwards compatibility
            return {
                'markers': [],
                'samples': [],
                'chromosomes': [],
                'cM': [],
                'Mb': [],
                'allele_map': {},
                'founders': [],
                'het_code': 2,
                'unk_code': 3,
                'dataset_name': dataset_id,
                'cross_type': '',
                'mat_allele': None,
                'pat_allele': None,
            }

    def find_nearest_full_snapshot(self, dataset_id: str, target_version: int) -> Optional[int]:
        """Find nearest full snapshot <= target."""
        for v in range(target_version, 0, -1):
            version_data = self.get_version(dataset_id, v)
            if version_data and version_data.storage_type == 'full':
                return v
        return None

    def get_current_version(self, dataset_id: str) -> Tuple[int, str]:
        """Get current version and hash."""
        with self.env.begin() as txn:
            geno_db = self.get_db(txn, self.DB_GENOTYPES)
            value = txn.get(self.make_info_key(dataset_id), db=geno_db)
            if not value:
                # Friendly error: suggest what the user can do
                all_datasets = []
                cursor = txn.cursor(db=geno_db)
                for key, _ in cursor:
                    all_datasets.append(key.decode('utf-8'))
                available = f" Available datasets: {', '.join(sorted(all_datasets))}" if all_datasets else " No datasets exist in this store."
                raise ValueError(
                    f"Dataset '{dataset_id}' not found in this store.{available}\n"
                    f"  - Check the dataset ID spelling (case-sensitive).\n"
                    f"  - Use 'list-datasets' to see all available datasets.\n"
                    f"  - Use 'import-genotype' to add a new dataset."
                )

            data = json.loads(value.decode('utf-8'))
            return data['current_version'], data['current_hash']

    def list_versions(self, dataset_id: str) -> List[Dict]:
        """List all versions."""
        versions = []
        prefix = f"{dataset_id}:".encode('utf-8')

        with self.env.begin() as txn:
            db = self.env.open_db(self.DB_MATRIX_HISTORY, txn=txn, create=False)
            cursor = txn.cursor(db=db)
            if cursor.set_range(prefix):
                for key, value in cursor:
                    if not key.startswith(prefix):
                        break
                    # Skip payload and metadata keys
                    if key.endswith(b':payload') or key.endswith(b':metadata'):
                        continue
                    versions.append(json.loads(value.decode('utf-8')))

        return sorted(versions, key=lambda x: x['matrix_version'])

    def verify_dataset(self, dataset_id: str) -> Tuple[bool, List[str]]:
        """Verify hash chain by recomputing all hashes from payloads."""
        errors = []
        versions = self.list_versions(dataset_id)

        if not versions:
            return False, ["No versions found"]

        prev_hash = None
        for i, v in enumerate(versions):
            if v['matrix_version'] != i + 1:
                errors.append(f"Version mismatch at {i}: expected {i+1}, got {v['matrix_version']}")

            if i == 0:
                if v['prev_matrix_hash'] is not None:
                    errors.append("v1 should have no prev_hash")
            else:
                if v['prev_matrix_hash'] != prev_hash:
                    errors.append(f"Hash chain broken at v{v['matrix_version']}")

            version_data = self.get_version(dataset_id, v['matrix_version'])
            computed_hash = compute_matrix_hash(version_data.payload, version_data.prev_matrix_hash)
            if computed_hash != v['matrix_hash']:
                errors.append(f"Hash mismatch at v{v['matrix_version']}")

            prev_hash = v['matrix_hash']

        return len(errors) == 0, errors

    def verify_dataset_fast(self, dataset_id: str) -> Tuple[bool, List[str]]:
        """Fast verify: check hash chain linkage without payload recomputation.

        Detects missing versions and broken chain links.
        Does NOT detect corrupted payloads (use verify_dataset for that).
        """
        errors = []
        versions = self.list_versions(dataset_id)

        if not versions:
            return False, ["No versions found"]

        prev_hash = None
        for i, v in enumerate(versions):
            if v['matrix_version'] != i + 1:
                errors.append(f"Version mismatch at {i}: expected {i+1}, got {v['matrix_version']}")

            if i == 0:
                if v['prev_matrix_hash'] is not None:
                    errors.append("v1 should have no prev_hash")
            else:
                if v['prev_matrix_hash'] != prev_hash:
                    errors.append(f"Hash chain broken at v{v['matrix_version']}")

            prev_hash = v['matrix_hash']

        return len(errors) == 0, errors

    def close(self):
        self.env.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
