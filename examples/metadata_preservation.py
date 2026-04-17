"""
Example: Metadata Preservation on Full Snapshots

This example demonstrates that when update-genotype creates a full snapshot
(shape change or scheduled interval), the metadata from the new .geno file
is preserved and available during reconstruction.
"""

import tempfile
import shutil
import numpy as np
from geno_storage.matrix_store import MatrixStore
from geno_storage.geno_parser import GenotypeMatrix


def main():
    temp_dir = tempfile.mkdtemp()
    try:
        store = MatrixStore(temp_dir)

        # v1: Initial import
        v1_matrix = GenotypeMatrix(
            matrix=np.array([
                [0, 0, 1],
                [1, 2, 0],
                [2, 0, 1]
            ], dtype=np.uint8),
            markers=["rs001", "rs002", "rs003"],
            samples=["BXD1", "BXD2", "BXD5"],
            chromosomes=["1", "1", "2"],
            positions=[1.5, 2.5, 0.0],
            allele_map={"B": 0, "D": 1},
            founders=["B", "D"],
            het_code=2,
            unk_code=3,
            dataset_name="BXD",
            cross_type="riset"
        )
        store.store_initial("BXD", v1_matrix, author="import", reason="Initial import")
        print("v1 stored: 3 markers, 3 samples")

        # v2-v3: Deltas (metadata from v1 reused)
        for i in range(2, 4):
            new_data = v1_matrix.matrix.copy()
            new_data[0, 0] = i
            store.store_update("BXD", new_data, "test", f"v{i} delta")
        print("v2-v3 stored: deltas, metadata from v1")

        # v4: Full snapshot with NEW metadata (simulating new .geno file)
        v4_matrix = GenotypeMatrix(
            matrix=np.array([
                [0, 0, 1, 1],
                [1, 2, 0, 1],
                [2, 0, 1, 0],
                [0, 1, 2, 2]
            ], dtype=np.uint8),
            markers=["rs001", "rs002", "rs003", "rs004"],  # NEW marker added
            samples=["BXD1", "BXD2", "BXD5", "BXD6"],     # NEW sample added
            chromosomes=["1", "1", "2", "X"],
            positions=[10.0, 20.0, 30.0, 40.0],
            allele_map={"A": 0, "T": 1},                   # Different encoding
            founders=["A", "T"],
            het_code=2,
            unk_code=3,
            dataset_name="BXD_Updated",
            cross_type="f2"
        )
        store.store_update("BXD", v4_matrix, "test", "v4 full with new metadata")
        print("v4 stored: full snapshot with 4 markers, 4 samples, NEW metadata")

        # v5: Delta (metadata from v4 reused)
        new_data = v4_matrix.matrix.copy()
        new_data[0, 0] = 5
        store.store_update("BXD", new_data, "test", "v5 delta")
        print("v5 stored: delta, metadata from v4")

        # Reconstruct and verify metadata at each version
        print("\n--- Reconstruction Results ---")

        recon_v1 = store.get_matrix("BXD", 1)
        print(f"v1: shape={recon_v1.matrix.shape}, markers={recon_v1.markers}, samples={recon_v1.samples}")

        recon_v3 = store.get_matrix("BXD", 3)
        print(f"v3: shape={recon_v3.matrix.shape}, markers={recon_v3.markers}, samples={recon_v3.samples}")

        recon_v4 = store.get_matrix("BXD", 4)
        print(f"v4: shape={recon_v4.matrix.shape}, markers={recon_v4.markers}, samples={recon_v4.samples}")
        print(f"     positions={recon_v4.positions}, allele_map={recon_v4.allele_map}")
        print(f"     dataset_name={recon_v4.dataset_name}, cross_type={recon_v4.cross_type}")

        recon_v5 = store.get_matrix("BXD", 5)
        print(f"v5: shape={recon_v5.matrix.shape}, markers={recon_v5.markers}, samples={recon_v5.samples}")

        # Verify integrity
        print("\n--- Verification ---")
        valid, errors = store.verify_dataset("BXD")
        print(f"Hash chain valid: {valid}")
        if errors:
            print(f"Errors: {errors}")

        store.close()

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
