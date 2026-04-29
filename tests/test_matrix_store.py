"""
Tests for LMDB-based matrix storage with versioning.
"""

import pytest
import numpy as np
import tempfile
import shutil
import os
from pathlib import Path

from geno_storage.matrix_store import MatrixStore, MatrixVersion
from geno_storage.geno_parser import GenotypeMatrix


class TestMatrixStoreBasics:
    """Test basic MatrixStore operations."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def sample_matrix(self):
        """Create a sample genotype matrix."""
        return GenotypeMatrix(
            matrix=np.array([[0, 1, 2], [1, 0, 3]], dtype=np.uint8),
            markers=["m1", "m2"],
            samples=["s1", "s2", "s3"],
            chromosomes=["1", "1"],
            cM=[1.0, 2.0],
            Mb=[1.0, 2.0],
            allele_map={"B": 0, "D": 1},
            founders=["B", "D"],
            het_code=2,
            unk_code=3,
            dataset_name="TestDS"
        )

    def test_create_store(self, temp_db):
        """Should create store without error."""
        store = MatrixStore(temp_db)
        assert store is not None
        store.close()

    def test_context_manager(self, temp_db):
        """Should work as context manager."""
        with MatrixStore(temp_db) as store:
            assert store is not None

    def test_store_initial(self, temp_db, sample_matrix):
        """Should store initial matrix."""
        store = MatrixStore(temp_db)

        version = store.store_initial("test_ds", sample_matrix)

        assert version.matrix_version == 1
        assert version.storage_type == "full"
        assert version.prev_matrix_hash is None
        assert version.dataset_id == "test_ds"

        store.close()

    def test_get_version(self, temp_db, sample_matrix):
        """Should retrieve stored version."""
        store = MatrixStore(temp_db)

        store.store_initial("test_ds", sample_matrix)
        retrieved = store.get_version("test_ds", 1)

        assert retrieved is not None
        assert retrieved.matrix_version == 1
        assert retrieved.nrows == 2
        assert retrieved.ncols == 3

        store.close()

    def test_get_current_version(self, temp_db, sample_matrix):
        """Should get current version info."""
        store = MatrixStore(temp_db)

        store.store_initial("test_ds", sample_matrix)
        version, hash_str = store.get_current_version("test_ds")

        assert version == 1
        assert hash_str is not None
        assert len(hash_str) == 64  # SHA256 hex

        store.close()

    def test_version_not_found(self, temp_db):
        """Should return None for non-existent version."""
        store = MatrixStore(temp_db)

        result = store.get_version("nonexistent", 1)
        assert result is None

        store.close()


class TestMatrixVersioning:
    """Test matrix versioning and updates."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def base_matrix(self):
        """Create base genotype matrix."""
        return GenotypeMatrix(
            matrix=np.zeros((5, 5), dtype=np.uint8),
            markers=[f"m{i}" for i in range(5)],
            samples=[f"s{i}" for i in range(5)],
            chromosomes=["1"] * 5,
            cM=[float(i) for i in range(5)],
            Mb=[float(i) for i in range(5)],
            allele_map={"B": 0, "D": 1},
            founders=["B", "D"],
            het_code=2,
            unk_code=3
        )

    def test_update_creates_version_2(self, temp_db, base_matrix):
        """Should create version 2 on update."""
        store = MatrixStore(temp_db)

        # Store initial
        store.store_initial("test_ds", base_matrix)

        # Update
        new_matrix = base_matrix.matrix.copy()
        new_matrix[0, 0] = 1

        version = store.store_update("test_ds", new_matrix, "test", "Update v2")

        assert version.matrix_version == 2
        assert version.prev_matrix_hash is not None
        assert version.reason == "Update v2"

        store.close()

    def test_multiple_updates(self, temp_db, base_matrix):
        """Should handle multiple sequential updates."""
        store = MatrixStore(temp_db)

        store.store_initial("test_ds", base_matrix)

        # Create 5 versions
        for i in range(2, 6):
            new_matrix = np.full((5, 5), i, dtype=np.uint8)
            store.store_update("test_ds", new_matrix, "test", f"Update v{i}")

        # Check current version
        version, _ = store.get_current_version("test_ds")
        assert version == 5

        store.close()

    def test_list_versions(self, temp_db, base_matrix):
        """Should list all versions."""
        store = MatrixStore(temp_db)

        store.store_initial("test_ds", base_matrix)
        new_matrix = np.ones((5, 5), dtype=np.uint8)
        store.store_update("test_ds", new_matrix, "test", "v2")

        versions = store.list_versions("test_ds")

        assert len(versions) == 2
        assert versions[0]["matrix_version"] == 1
        assert versions[1]["matrix_version"] == 2

        store.close()

    def test_full_snapshot_interval(self, temp_db, base_matrix):
        """Should store full snapshots at intervals."""
        store = MatrixStore(temp_db)
        store.FULL_SNAPSHOT_INTERVAL = 3  # Every 3 versions

        store.store_initial("test_ds", base_matrix)

        # v2, v3 should be deltas
        for i in range(2, 4):
            new_matrix = np.full((5, 5), i, dtype=np.uint8)
            store.store_update("test_ds", new_matrix, "test", f"v{i}")

        # v4 should be full (interval hit)
        new_matrix = np.full((5, 5), 4, dtype=np.uint8)
        v4 = store.store_update("test_ds", new_matrix, "test", "v4")

        assert v4.storage_type == "full"

        store.close()


class TestMatrixReconstruction:
    """Test matrix reconstruction from ledger."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def base_matrix(self):
        """Create base genotype matrix."""
        return GenotypeMatrix(
            matrix=np.zeros((10, 10), dtype=np.uint8),
            markers=[f"m{i}" for i in range(10)],
            samples=[f"s{i}" for i in range(10)],
            chromosomes=["1"] * 10,
            cM=[float(i) for i in range(10)],
            Mb=[float(i) for i in range(10)],
            allele_map={"B": 0, "D": 1},
            founders=["B", "D"],
            het_code=2,
            unk_code=3
        )

    def test_reconstruct_current(self, temp_db, base_matrix):
        """Should reconstruct current version."""
        store = MatrixStore(temp_db)

        store.store_initial("test_ds", base_matrix)
        new_matrix = np.ones((10, 10), dtype=np.uint8)
        store.store_update("test_ds", new_matrix, "test", "v2")

        reconstructed = store.get_matrix("test_ds")

        # Now returns GenotypeMatrix, check matrix data
        np.testing.assert_array_equal(reconstructed.matrix, new_matrix)
        # Also verify metadata preserved
        assert reconstructed.markers == base_matrix.markers
        assert reconstructed.samples == base_matrix.samples

        store.close()

    def test_reconstruct_specific_version(self, temp_db, base_matrix):
        """Should reconstruct specific historical version."""
        store = MatrixStore(temp_db)

        store.store_initial("test_ds", base_matrix)

        v2_matrix = np.ones((10, 10), dtype=np.uint8)
        store.store_update("test_ds", v2_matrix, "test", "v2")

        v3_matrix = np.full((10, 10), 2, dtype=np.uint8)
        store.store_update("test_ds", v3_matrix, "test", "v3")

        # Reconstruct v1
        v1 = store.get_matrix("test_ds", 1)
        np.testing.assert_array_equal(v1.matrix, base_matrix.matrix)

        # Reconstruct v2
        v2 = store.get_matrix("test_ds", 2)
        np.testing.assert_array_equal(v2.matrix, v2_matrix)

        store.close()

    def test_reconstruct_from_deltas(self, temp_db, base_matrix):
        """Should reconstruct by replaying deltas."""
        store = MatrixStore(temp_db)
        store.FULL_SNAPSHOT_INTERVAL = 10  # Force deltas

        store.store_initial("test_ds", base_matrix)

        # Create sparse updates - each builds on previous
        current = base_matrix.matrix.copy()
        for i in range(2, 6):
            current = current.copy()
            current[i-1, i-1] = i
            store.store_update("test_ds", current, "test", f"v{i}")

        # Reconstruct v5
        reconstructed = store.get_matrix("test_ds", 5)

        expected = base_matrix.matrix.copy()
        expected[1, 1] = 2
        expected[2, 2] = 3
        expected[3, 3] = 4
        expected[4, 4] = 5

        np.testing.assert_array_equal(reconstructed.matrix, expected)

        store.close()

    def test_reconstruct_missing_version(self, temp_db, base_matrix):
        """Should raise error for missing version."""
        store = MatrixStore(temp_db)

        store.store_initial("test_ds", base_matrix)

        with pytest.raises(ValueError):
            store.get_matrix("test_ds", 99)

        store.close()


class TestHashVerification:
    """Test hash chain verification."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def sample_matrix(self):
        """Create sample matrix."""
        return GenotypeMatrix(
            matrix=np.zeros((5, 5), dtype=np.uint8),
            markers=[f"m{i}" for i in range(5)],
            samples=[f"s{i}" for i in range(5)],
            chromosomes=["1"] * 5,
            cM=[float(i) for i in range(5)],
            Mb=[float(i) for i in range(5)],
            allele_map={"B": 0, "D": 1},
            founders=["B", "D"],
            het_code=2,
            unk_code=3
        )

    def test_verify_valid_dataset(self, temp_db, sample_matrix):
        """Should verify valid dataset."""
        store = MatrixStore(temp_db)

        store.store_initial("test_ds", sample_matrix)
        new_matrix = np.ones((5, 5), dtype=np.uint8)
        store.store_update("test_ds", new_matrix, "test", "v2")

        valid, errors = store.verify_dataset("test_ds")

        assert valid is True
        assert len(errors) == 0

        store.close()

    def test_verify_nonexistent_dataset(self, temp_db):
        """Should fail for non-existent dataset."""
        store = MatrixStore(temp_db)

        valid, errors = store.verify_dataset("nonexistent")

        assert valid is False
        assert len(errors) > 0

        store.close()

    def test_hash_chain_integrity(self, temp_db, sample_matrix):
        """Hash chain should be maintained correctly."""
        store = MatrixStore(temp_db)

        v1 = store.store_initial("test_ds", sample_matrix)
        assert v1.prev_matrix_hash is None

        new_matrix = np.ones((5, 5), dtype=np.uint8)
        v2 = store.store_update("test_ds", new_matrix, "test", "v2")
        assert v2.prev_matrix_hash == v1.matrix_hash

        new_matrix2 = np.full((5, 5), 2, dtype=np.uint8)
        v3 = store.store_update("test_ds", new_matrix2, "test", "v3")
        assert v3.prev_matrix_hash == v2.matrix_hash

        store.close()

    def test_hash_uniqueness(self, temp_db, sample_matrix):
        """Different data should produce different hashes."""
        store = MatrixStore(temp_db)

        v1 = store.store_initial("test_ds", sample_matrix)

        new_matrix = np.ones((5, 5), dtype=np.uint8)
        v2 = store.store_update("test_ds", new_matrix, "test", "v2")

        assert v1.matrix_hash != v2.matrix_hash

        store.close()


class TestMultiDataset:
    """Test handling multiple datasets in one store."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_multiple_datasets(self, temp_db):
        """Should handle multiple independent datasets."""
        store = MatrixStore(temp_db)

        matrix1 = GenotypeMatrix(
            matrix=np.zeros((5, 5), dtype=np.uint8),
            markers=[f"m{i}" for i in range(5)],
            samples=[f"s{i}" for i in range(5)],
            chromosomes=["1"] * 5,
            cM=[float(i) for i in range(5)],
            Mb=[float(i) for i in range(5)],
            allele_map={"B": 0, "D": 1},
            founders=["B", "D"],
            het_code=2,
            unk_code=3
        )

        matrix2 = GenotypeMatrix(
            matrix=np.ones((3, 3), dtype=np.uint8),
            markers=[f"mx{i}" for i in range(3)],
            samples=[f"sx{i}" for i in range(3)],
            chromosomes=["X"] * 3,
            cM=[float(i) for i in range(3)],
            Mb=[float(i) for i in range(3)],
            allele_map={"A": 0, "B": 1},
            founders=["A", "B"],
            het_code=2,
            unk_code=3
        )

        store.store_initial("dataset_a", matrix1)
        store.store_initial("dataset_b", matrix2)

        # Verify both exist independently
        v1_a, _ = store.get_current_version("dataset_a")
        v1_b, _ = store.get_current_version("dataset_b")

        assert v1_a == 1
        assert v1_b == 1

        # Verify different matrices
        recon_a = store.get_matrix("dataset_a")
        recon_b = store.get_matrix("dataset_b")

        assert recon_a.matrix.shape == (5, 5)
        assert recon_b.matrix.shape == (3, 3)

        store.close()

    def test_dataset_isolation(self, temp_db):
        """Datasets should be isolated from each other."""
        store = MatrixStore(temp_db)

        matrix = GenotypeMatrix(
            matrix=np.zeros((5, 5), dtype=np.uint8),
            markers=[f"m{i}" for i in range(5)],
            samples=[f"s{i}" for i in range(5)],
            chromosomes=["1"] * 5,
            cM=[float(i) for i in range(5)],
            Mb=[float(i) for i in range(5)],
            allele_map={"B": 0, "D": 1},
            founders=["B", "D"],
            het_code=2,
            unk_code=3
        )

        store.store_initial("ds1", matrix)
        store.store_update("ds1", np.ones((5, 5), dtype=np.uint8), "test", "v2")

        store.store_initial("ds2", matrix)

        # ds1 should be at v2, ds2 at v1
        v1, _ = store.get_current_version("ds1")
        v2, _ = store.get_current_version("ds2")

        assert v1 == 2
        assert v2 == 1

        store.close()


class TestErrorHandling:
    """Test error handling."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_update_nonexistent_dataset(self, temp_db):
        """Should raise error when updating non-existent dataset."""
        store = MatrixStore(temp_db)

        with pytest.raises(ValueError):
            store.store_update("nonexistent", np.zeros((5, 5)), "test", "update")

        store.close()

    def test_shape_mismatch_update(self, temp_db):
        """Should handle shape changes (stores full snapshot)."""
        store = MatrixStore(temp_db)

        matrix = GenotypeMatrix(
            matrix=np.zeros((5, 5), dtype=np.uint8),
            markers=[f"m{i}" for i in range(5)],
            samples=[f"s{i}" for i in range(5)],
            chromosomes=["1"] * 5,
            cM=[float(i) for i in range(5)],
            Mb=[float(i) for i in range(5)],
            allele_map={"B": 0, "D": 1},
            founders=["B", "D"],
            het_code=2,
            unk_code=3
        )

        store.store_initial("test_ds", matrix)

        # Different shape - should work as full snapshot
        new_matrix = np.ones((5, 6), dtype=np.uint8)
        v2 = store.store_update("test_ds", new_matrix, "test", "shape change")

        assert v2 is not None

        store.close()


class TestFullSnapshotMetadata:
    """Test that metadata is written on full snapshots when GenotypeMatrix is provided."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def base_matrix(self):
        """Create base genotype matrix with rich metadata."""
        return GenotypeMatrix(
            matrix=np.zeros((5, 5), dtype=np.uint8),
            markers=["m1", "m2", "m3", "m4", "m5"],
            samples=["s1", "s2", "s3", "s4", "s5"],
            chromosomes=["1", "1", "2", "2", "X"],
            cM=[1.0, 2.0, 3.0, 4.0, 5.0],
            Mb=[1.0, 2.0, 3.0, 4.0, 5.0],
            allele_map={"B": 0, "D": 1},
            founders=["B", "D"],
            het_code=2,
            unk_code=3,
            dataset_name="TestDS",
            cross_type="riset",
            mat_allele="B",
            pat_allele="D"
        )

    def test_full_snapshot_with_genotype_matrix_writes_metadata(self, temp_db, base_matrix):
        """store_update with GenotypeMatrix + full snapshot should write :metadata."""
        store = MatrixStore(temp_db)
        store.FULL_SNAPSHOT_INTERVAL = 1  # Force full snapshot on every update

        store.store_initial("test_ds", base_matrix)

        # Create updated GenotypeMatrix with different values but same metadata
        updated = GenotypeMatrix(
            matrix=np.ones((5, 5), dtype=np.uint8),
            markers=["m1", "m2", "m3", "m4", "m5"],
            samples=["s1", "s2", "s3", "s4", "s5"],
            chromosomes=["1", "1", "2", "2", "X"],
            cM=[10.0, 20.0, 30.0, 40.0, 50.0],
            Mb=[10.0, 20.0, 30.0, 40.0, 50.0],
            allele_map={"B": 0, "D": 1},
            founders=["B", "D"],
            het_code=2,
            unk_code=3,
            dataset_name="TestDS",
            cross_type="riset",
            mat_allele="B",
            pat_allele="D"
        )

        v2 = store.store_update("test_ds", updated, "test", "v2 with metadata")
        assert v2.storage_type == "full"

        # Reconstruct v2 — should get metadata from v2 full snapshot
        reconstructed = store.get_matrix("test_ds", 2)
        assert reconstructed.markers == updated.markers
        assert reconstructed.samples == updated.samples
        assert reconstructed.cM == updated.cM
        assert reconstructed.Mb == updated.Mb
        assert reconstructed.chromosomes == updated.chromosomes
        assert reconstructed.allele_map == updated.allele_map
        assert reconstructed.founders == updated.founders
        assert reconstructed.cross_type == updated.cross_type

        store.close()

    def test_full_snapshot_with_numpy_array_no_metadata(self, temp_db, base_matrix):
        """store_update with numpy array + full snapshot should not write :metadata."""
        store = MatrixStore(temp_db)
        store.FULL_SNAPSHOT_INTERVAL = 1  # Force full snapshot

        store.store_initial("test_ds", base_matrix)

        # Pass plain numpy array (as CLI currently does)
        new_matrix = np.ones((5, 5), dtype=np.uint8)
        v2 = store.store_update("test_ds", new_matrix, "test", "v2 no metadata")
        assert v2.storage_type == "full"

        # Reconstruct v2 — should fallback to empty metadata (backward compatible)
        reconstructed = store.get_matrix("test_ds", 2)
        np.testing.assert_array_equal(reconstructed.matrix, new_matrix)
        assert reconstructed.markers == []  # Fallback empty
        assert reconstructed.samples == []  # Fallback empty

        store.close()

    def test_delta_does_not_write_metadata(self, temp_db, base_matrix):
        """store_update with delta should not write or overwrite :metadata."""
        store = MatrixStore(temp_db)
        store.FULL_SNAPSHOT_INTERVAL = 10  # Force delta

        store.store_initial("test_ds", base_matrix)

        updated = GenotypeMatrix(
            matrix=np.zeros((5, 5), dtype=np.uint8),
            markers=["new_m1", "new_m2", "new_m3", "new_m4", "new_m5"],
            samples=["new_s1", "new_s2", "new_s3", "new_s4", "new_s5"],
            chromosomes=["1", "1", "2", "2", "X"],
            cM=[100.0, 200.0, 300.0, 400.0, 500.0],
            Mb=[100.0, 200.0, 300.0, 400.0, 500.0],
            allele_map={"A": 0, "C": 1},
            founders=["A", "C"],
            het_code=2,
            unk_code=3,
            dataset_name="UpdatedDS",
            cross_type="hs"
        )

        # Make a small change so delta is used
        updated.matrix[2, 2] = 1

        v2 = store.store_update("test_ds", updated, "test", "v2 delta")
        assert v2.storage_type == "delta"

        # Reconstruct v2 — metadata should come from v1, not v2
        reconstructed = store.get_matrix("test_ds", 2)
        assert reconstructed.markers == base_matrix.markers  # From v1
        assert reconstructed.samples == base_matrix.samples  # From v1
        assert reconstructed.allele_map == base_matrix.allele_map  # From v1

        store.close()

    def test_scheduled_full_snapshot_metadata(self, temp_db, base_matrix):
        """Scheduled full snapshot at interval should write :metadata when GenotypeMatrix provided."""
        store = MatrixStore(temp_db)
        store.FULL_SNAPSHOT_INTERVAL = 3  # v4 will be full

        store.store_initial("test_ds", base_matrix)

        # v2, v3: deltas with numpy array
        for i in range(2, 4):
            new_matrix = np.full((5, 5), i, dtype=np.uint8)
            store.store_update("test_ds", new_matrix, "test", f"v{i}")

        # v4: scheduled full with GenotypeMatrix
        updated = GenotypeMatrix(
            matrix=np.full((5, 5), 4, dtype=np.uint8),
            markers=["v4_m1", "v4_m2", "v4_m3", "v4_m4", "v4_m5"],
            samples=["v4_s1", "v4_s2", "v4_s3", "v4_s4", "v4_s5"],
            chromosomes=["1", "2", "3", "4", "5"],
            cM=[10.0, 20.0, 30.0, 40.0, 50.0],
            Mb=[10.0, 20.0, 30.0, 40.0, 50.0],
            allele_map={"A": 0, "T": 1},
            founders=["A", "T"],
            het_code=2,
            unk_code=3,
            dataset_name="V4Dataset",
            cross_type="f2"
        )

        v4 = store.store_update("test_ds", updated, "test", "v4 full")
        assert v4.storage_type == "full"

        # Reconstruct v4 — should get metadata from v4 full snapshot
        reconstructed = store.get_matrix("test_ds", 4)
        assert reconstructed.markers == updated.markers
        assert reconstructed.samples == updated.samples
        assert reconstructed.chromosomes == updated.chromosomes
        assert reconstructed.cM == updated.cM
        assert reconstructed.Mb == updated.Mb
        assert reconstructed.allele_map == updated.allele_map
        assert reconstructed.founders == updated.founders
        assert reconstructed.dataset_name == updated.dataset_name
        assert reconstructed.cross_type == updated.cross_type

        # v5: delta — metadata should still come from v4
        v5_matrix = np.full((5, 5), 5, dtype=np.uint8)
        store.store_update("test_ds", v5_matrix, "test", "v5 delta")

        recon_v5 = store.get_matrix("test_ds", 5)
        assert recon_v5.markers == updated.markers  # From v4

        store.close()


class TestFastVerification:
    """Test fast verify (chain linkage only, no payload hashing)."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def sample_matrix(self):
        """Create sample matrix."""
        return GenotypeMatrix(
            matrix=np.zeros((5, 5), dtype=np.uint8),
            markers=[f"m{i}" for i in range(5)],
            samples=[f"s{i}" for i in range(5)],
            chromosomes=["1"] * 5,
            cM=[float(i) for i in range(5)],
            Mb=[float(i) for i in range(5)],
            allele_map={"B": 0, "D": 1},
            founders=["B", "D"],
            het_code=2,
            unk_code=3
        )

    def test_fast_verify_valid_dataset(self, temp_db, sample_matrix):
        """Fast verify should pass for valid dataset."""
        store = MatrixStore(temp_db)

        store.store_initial("test_ds", sample_matrix)
        new_matrix = np.ones((5, 5), dtype=np.uint8)
        store.store_update("test_ds", new_matrix, "test", "v2")

        valid, errors = store.verify_dataset_fast("test_ds")

        assert valid is True
        assert len(errors) == 0

        store.close()

    def test_fast_verify_nonexistent_dataset(self, temp_db):
        """Fast verify should fail for non-existent dataset."""
        store = MatrixStore(temp_db)

        valid, errors = store.verify_dataset_fast("nonexistent")

        assert valid is False
        assert len(errors) > 0

        store.close()

    def test_fast_verify_detects_broken_chain(self, temp_db, sample_matrix):
        """Fast verify should detect broken prev_hash linkage."""
        import json
        store = MatrixStore(temp_db)

        store.store_initial("test_ds", sample_matrix)
        new_matrix = np.ones((5, 5), dtype=np.uint8)
        store.store_update("test_ds", new_matrix, "test", "v2")

        # Tamper with v2's prev_hash
        with store.env.begin(write=True) as txn:
            matrix_db = store.env.open_db(store.DB_MATRIX_HISTORY, txn=txn, create=False)
            key = store._make_history_key("test_ds", 2)
            value = txn.get(key, db=matrix_db)
            data = json.loads(value.decode('utf-8'))
            data['prev_matrix_hash'] = 'tampered_hash_12345'
            txn.put(key, json.dumps(data).encode('utf-8'), db=matrix_db)

        # Full verify: catches both chain break AND hash mismatch
        valid_full, errors_full = store.verify_dataset("test_ds")
        assert valid_full is False
        assert any("Hash chain broken" in e for e in errors_full)
        assert any("Hash mismatch" in e for e in errors_full)

        # Fast verify: catches only chain break
        valid_fast, errors_fast = store.verify_dataset_fast("test_ds")
        assert valid_fast is False
        assert any("Hash chain broken" in e for e in errors_fast)
        assert not any("Hash mismatch" in e for e in errors_fast)

        store.close()

    def test_fast_verify_missed_payload_corruption(self, temp_db, sample_matrix):
        """Fast verify should miss payload-only corruption (by design)."""
        store = MatrixStore(temp_db)

        store.store_initial("test_ds", sample_matrix)
        new_matrix = np.ones((5, 5), dtype=np.uint8)
        store.store_update("test_ds", new_matrix, "test", "v2")

        # Tamper with v2's payload only
        with store.env.begin(write=True) as txn:
            matrix_db = store.env.open_db(store.DB_MATRIX_HISTORY, txn=txn, create=False)
            key = store._make_history_key("test_ds", 2)
            payload_key = key + b':payload'
            payload = txn.get(payload_key, db=matrix_db)
            tampered = payload[:10] + bytes([payload[10] ^ 0xFF]) + payload[11:]
            txn.put(payload_key, tampered, db=matrix_db)

        # Fast verify: misses payload-only corruption
        valid_fast, errors_fast = store.verify_dataset_fast("test_ds")
        assert valid_fast is True
        assert len(errors_fast) == 0

        # Full verify: catches payload corruption
        valid_full, errors_full = store.verify_dataset("test_ds")
        assert valid_full is False
        assert any("Hash mismatch" in e for e in errors_full)

        store.close()

    def test_fast_verify_matches_full_on_valid_chain(self, temp_db, sample_matrix):
        """Both verify modes should agree on valid chains."""
        store = MatrixStore(temp_db)

        store.store_initial("test_ds", sample_matrix)

        for i in range(2, 8):
            new_matrix = np.full((5, 5), i, dtype=np.uint8)
            store.store_update("test_ds", new_matrix, "test", f"v{i}")

        valid_full, _ = store.verify_dataset("test_ds")
        valid_fast, _ = store.verify_dataset_fast("test_ds")

        assert valid_full is True
        assert valid_fast is True
        assert valid_full == valid_fast

        store.close()


class TestFriendlyErrorMessages:
    """Test that error messages are actionable and helpful."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def sample_matrix(self):
        """Create sample matrix."""
        return GenotypeMatrix(
            matrix=np.zeros((5, 5), dtype=np.uint8),
            markers=[f"m{i}" for i in range(5)],
            samples=[f"s{i}" for i in range(5)],
            chromosomes=["1"] * 5,
            cM=[float(i) for i in range(5)],
            Mb=[float(i) for i in range(5)],
            allele_map={"B": 0, "D": 1},
            founders=["B", "D"],
            het_code=2,
            unk_code=3
        )

    def test_missing_dataset_error_suggests_list_datasets(self, temp_db):
        """Missing dataset error should suggest using list-datasets."""
        store = MatrixStore(temp_db)

        with pytest.raises(ValueError) as exc_info:
            store.get_current_version("nonexistent")

        msg = str(exc_info.value)
        assert "list-datasets" in msg or "list_datasets" in msg
        assert "not found" in msg.lower()

    def test_missing_dataset_error_lists_available(self, temp_db, sample_matrix):
        """Missing dataset error should list available datasets."""
        store = MatrixStore(temp_db)
        store.store_initial("BXD", sample_matrix)
        store.store_initial("HSRats", sample_matrix)

        with pytest.raises(ValueError) as exc_info:
            store.get_current_version("MissingDS")

        msg = str(exc_info.value)
        assert "BXD" in msg
        assert "HSRats" in msg
        assert "MissingDS" in msg

    def test_reconstruct_missing_version_suggests_verify(self, temp_db, sample_matrix):
        """Reconstruction error should suggest using verify."""
        store = MatrixStore(temp_db)
        store.store_initial("test_ds", sample_matrix)

        # Delete the full snapshot to simulate corruption
        import json
        with store.env.begin(write=True) as txn:
            matrix_db = store.env.open_db(store.DB_MATRIX_HISTORY, txn=txn, create=False)
            key = store._make_history_key("test_ds", 1)
            txn.delete(key + b':payload', db=matrix_db)

        with pytest.raises(ValueError) as exc_info:
            store.get_matrix("test_ds", 1)

        msg = str(exc_info.value)
        assert "verify" in msg.lower()
        assert "missing" in msg.lower()

    def test_reconstruct_missing_delta_suggests_verify(self, temp_db, sample_matrix):
        """Missing delta error should explain which versions are needed."""
        store = MatrixStore(temp_db)
        store.FULL_SNAPSHOT_INTERVAL = 10  # Force delta for v2
        store.store_initial("test_ds", sample_matrix)

        new_matrix = sample_matrix.matrix.copy()
        new_matrix[0, 0] = 1
        store.store_update("test_ds", new_matrix, "test", "v2")

        # Delete the delta payload to simulate corruption
        with store.env.begin(write=True) as txn:
            matrix_db = store.env.open_db(store.DB_MATRIX_HISTORY, txn=txn, create=False)
            key = store._make_history_key("test_ds", 2)
            txn.delete(key + b':payload', db=matrix_db)

        with pytest.raises(ValueError) as exc_info:
            store.get_matrix("test_ds", 2)

        msg = str(exc_info.value)
        assert "delta" in msg.lower()
        assert "verify" in msg.lower()


class TestMatrixVersionDataclass:
    """Test MatrixVersion dataclass."""

    def test_to_dict(self):
        """Should convert to dictionary."""
        version = MatrixVersion(
            dataset_id="test",
            matrix_version=1,
            matrix_hash="abc123",
            prev_matrix_hash=None,
            storage_type="full",
            payload=b"data",
            timestamp="2024-01-01T00:00:00Z",
            reason="test",
            author="pytest",
            nrows=10,
            ncols=20,
            dtype="uint8"
        )

        d = version.to_dict()

        assert d["dataset_id"] == "test"
        assert d["matrix_version"] == 1
        assert d["matrix_hash"] == "abc123"
        assert "payload" not in d  # Payload excluded

    def test_from_dict(self):
        """Should create from dictionary."""
        data = {
            "dataset_id": "test",
            "matrix_version": 1,
            "matrix_hash": "abc123",
            "prev_matrix_hash": None,
            "storage_type": "full",
            "timestamp": "2024-01-01T00:00:00Z",
            "reason": "test",
            "author": "pytest",
            "nrows": 10,
            "ncols": 20,
            "dtype": "uint8"
        }
        payload = b"test_payload"

        version = MatrixVersion.from_dict(data, payload)

        assert version.dataset_id == "test"
        assert version.payload == payload
