"""
Example: CLI Update Workflow with Metadata Preservation

This example simulates what the update-genotype CLI does:
1. Parse a .geno file (full GenotypeMatrix with metadata)
2. Pass the full GenotypeMatrix to store_update()
3. Verify metadata is preserved on full snapshots
"""

import tempfile
import shutil
import numpy as np
import os

from geno_storage.matrix_store import MatrixStore
from geno_storage.geno_parser import parse_genotype_file
from geno_storage.models import GenotypeMatrix


def create_geno_file(path, content):
    """Write genotype file."""
    with open(path, 'w') as f:
        f.write(content)


def main():
    temp_dir = tempfile.mkdtemp()
    try:
        # Create initial .geno file
        geno_v1 = os.path.join(temp_dir, "BXD_v1.geno")
        create_geno_file(geno_v1, """# File name: BXD_v1.geno
@name:BXD
@type:riset
@mat:B
@pat:D
@het:H
@unk:U
Chr\tLocus\tcM\tMb\tBXD1\tBXD2\tBXD3
1\trs001\t1.0\t5.0\tB\tB\tD
1\trs002\t2.0\t6.0\tD\tH\tU
2\trs003\t0.0\t10.0\tH\tU\tB
""")

        # Create updated .geno file (corrected values + new marker)
        geno_v2 = os.path.join(temp_dir, "BXD_v2.geno")
        create_geno_file(geno_v2, """# File name: BXD_v2.geno
@name:BXD_Corrected
@type:riset
@mat:B
@pat:D
@het:H
@unk:U
Chr\tLocus\tcM\tMb\tBXD1\tBXD2\tBXD3\tBXD5
1\trs001\t1.0\t5.0\tB\tD\tD\tB
1\trs002\t2.0\t6.0\tD\tB\tB\tH
1\trs003\t3.0\t7.0\tH\tB\tD\tU
2\trs004\t0.0\t10.0\tB\tD\tH\tB
""")

        lmdb_path = os.path.join(temp_dir, "lmdb_store")
        store = MatrixStore(lmdb_path)

        # Step 1: Import initial (like import-genotype CLI)
        print("=== Step 1: Import Initial ===")
        genotype_v1 = parse_genotype_file(geno_v1)
        print(f"Parsed: {genotype_v1.dataset_name}")
        print(f"  Markers: {genotype_v1.markers}")
        print(f"  Samples: {genotype_v1.samples}")
        print(f"  Shape: {genotype_v1.matrix.shape}")

        v1 = store.store_initial("BXD", genotype_v1, author="import", reason="Initial import")
        print(f"Stored as v1 ({v1.storage_type})")

        # Step 2: Update with new .geno (like update-genotype CLI)
        # KEY: Pass FULL GenotypeMatrix, not just .matrix
        print("\n=== Step 2: Update with New .geno ===")
        genotype_v2 = parse_genotype_file(geno_v2)
        print(f"Parsed: {genotype_v2.dataset_name}")
        print(f"  Markers: {genotype_v2.markers}")
        print(f"  Samples: {genotype_v2.samples}")
        print(f"  Shape: {genotype_v2.matrix.shape}")

        v2 = store.store_update("BXD", genotype_v2, author="qc_pipeline", reason="QC: added rs004, corrected errors")
        print(f"Stored as v2 ({v2.storage_type})")

        # Step 3: Reconstruct and verify metadata
        print("\n=== Step 3: Reconstruct ===")

        recon_v1 = store.get_matrix("BXD", 1)
        print(f"v1 reconstructed:")
        print(f"  Shape: {recon_v1.matrix.shape}")
        print(f"  Markers: {recon_v1.markers}")
        print(f"  Samples: {recon_v1.samples}")

        recon_v2 = store.get_matrix("BXD", 2)
        print(f"v2 reconstructed:")
        print(f"  Shape: {recon_v2.matrix.shape}")
        print(f"  Markers: {recon_v2.markers}")
        print(f"  Samples: {recon_v2.samples}")
        print(f"  cM: {recon_v2.cM}, Mb: {recon_v2.Mb}")
        print(f"  Founders: {recon_v2.founders}")
        print(f"  Cross type: {recon_v2.cross_type}")

        # Step 4: Verify integrity
        print("\n=== Step 4: Verify ===")
        valid, errors = store.verify_dataset("BXD")
        print(f"Hash chain valid: {valid}")
        if errors:
            print(f"Errors: {errors}")

        # Demonstrate: metadata from v2 is preserved because full snapshot
        assert recon_v2.markers == ["rs001", "rs002", "rs003", "rs004"], "Metadata preserved!"
        assert recon_v2.samples == ["BXD1", "BXD2", "BXD3", "BXD5"], "Metadata preserved!"
        print("\n✓ All metadata correctly preserved from updated .geno file")

        store.close()

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
