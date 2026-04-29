"""
Integration tests for the full workflow.

Tests the complete pipeline: parse .geno file -> store -> version -> reconstruct.
"""

import pytest
import numpy as np
import tempfile
import shutil
import os
from pathlib import Path

from geno_storage.geno_parser import parse_genotype_file, GenotypeMatrix
from geno_storage.matrix_store import MatrixStore
from geno_storage.delta import DeltaEncoder
from geno_storage.hashing import verify_hash_chain


class TestFullWorkflow:
    """Test complete workflow from parsing to reconstruction."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def sample_geno_file(self):
        """Create a sample genotype file."""
        content = """# File name: test.geno
# Description: Integration test file
@name:TestDataset
@type:riset
@mat:B
@pat:D
@het:H
@unk:U
Chr\tLocus\tcM\tMb\tBXD1\tBXD2\tBXD3\tBXD4\tBXD5
1\trs001\t1.0\t5.0\tB\tB\tD\tD\tH
1\trs002\t2.0\t6.0\tD\tH\tB\tU\tB
1\trs003\t3.0\t7.0\tH\tU\tH\tB\tD
2\trs004\t0.0\t10.0\tB\tD\tD\tH\tU
2\trs005\t1.0\t11.0\tU\tB\tU\tD\tB
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.geno', delete=False) as f:
            f.write(content)
            temp_path = f.name

        yield temp_path
        os.unlink(temp_path)

    def test_parse_and_store(self, temp_db, sample_geno_file):
        """Should parse genotype file and store in LMDB."""
        # Parse
        genotype = parse_genotype_file(sample_geno_file)

        assert genotype.dataset_name == "TestDataset"
        assert genotype.matrix.shape == (5, 5)  # 5 markers, 5 samples

        # Store
        store = MatrixStore(temp_db)
        version = store.store_initial("test_dataset", genotype)

        assert version.matrix_version == 1
        assert version.storage_type == "full"

        store.close()

    def test_store_multiple_versions(self, temp_db, sample_geno_file):
        """Should store and version updates."""
        # Parse and store initial
        genotype = parse_genotype_file(sample_geno_file)

        store = MatrixStore(temp_db)
        store.store_initial("test_dataset", genotype, reason="Initial import")

        # Simulate QC correction - update some values
        corrected_matrix = genotype.matrix.copy()
        corrected_matrix[0, 0] = 1  # Change B to D
        corrected_matrix[2, 2] = 0  # Change H to B

        v2 = store.store_update(
            "test_dataset",
            corrected_matrix,
            author="qc_pipeline",
            reason="QC correction: fix genotyping errors"
        )

        assert v2.matrix_version == 2
        assert v2.author == "qc_pipeline"
        assert "QC correction" in v2.reason

        # Add another version
        final_matrix = corrected_matrix.copy()
        final_matrix[4, 4] = 1  # Another change

        v3 = store.store_update(
            "test_dataset",
            final_matrix,
            author="analyst_1",
            reason="Manual curation"
        )

        assert v3.matrix_version == 3

        # Verify all versions exist
        versions = store.list_versions("test_dataset")
        assert len(versions) == 3

        store.close()

    def test_reconstruct_historical_versions(self, temp_db, sample_geno_file):
        """Should reconstruct any historical version."""
        genotype = parse_genotype_file(sample_geno_file)

        store = MatrixStore(temp_db)
        store.store_initial("test_dataset", genotype)

        # Create versions with known changes
        v1_matrix = genotype.matrix.copy()

        v2_matrix = v1_matrix.copy()
        v2_matrix[0, :] = 1  # Change entire first row to D
        store.store_update("test_dataset", v2_matrix, "test", "v2")

        v3_matrix = v2_matrix.copy()
        v3_matrix[:, 0] = 2  # Change entire first column to H
        store.store_update("test_dataset", v3_matrix, "test", "v3")

        # Reconstruct each version
        recon_v1 = store.get_matrix("test_dataset", 1)
        recon_v2 = store.get_matrix("test_dataset", 2)
        recon_v3 = store.get_matrix("test_dataset", 3)
        recon_current = store.get_matrix("test_dataset")

        # Verify (access .matrix for GenotypeMatrix)
        np.testing.assert_array_equal(recon_v1.matrix, v1_matrix)
        np.testing.assert_array_equal(recon_v2.matrix, v2_matrix)
        np.testing.assert_array_equal(recon_v3.matrix, v3_matrix)
        np.testing.assert_array_equal(recon_current.matrix, v3_matrix)

        store.close()

    def test_verification(self, temp_db, sample_geno_file):
        """Should verify hash chain integrity."""
        genotype = parse_genotype_file(sample_geno_file)

        store = MatrixStore(temp_db)
        store.store_initial("test_dataset", genotype)

        for i in range(2, 5):
            new_matrix = genotype.matrix.copy()
            new_matrix[i-1, i-1] = i
            store.store_update("test_dataset", new_matrix, "test", f"v{i}")

        # Verify dataset integrity
        valid, errors = store.verify_dataset("test_dataset")

        assert valid is True, f"Verification failed: {errors}"
        assert len(errors) == 0

        store.close()

    def test_list_versions_metadata(self, temp_db, sample_geno_file):
        """Should retrieve version metadata."""
        genotype = parse_genotype_file(sample_geno_file)

        store = MatrixStore(temp_db)
        store.store_initial("test_dataset", genotype, author="import", reason="Initial")

        store.store_update(
            "test_dataset",
            genotype.matrix.copy(),
            author="pipeline_v2",
            reason="QC pass"
        )

        versions = store.list_versions("test_dataset")

        assert len(versions) == 2
        assert versions[0]["author"] == "import"
        assert versions[1]["author"] == "pipeline_v2"
        assert "timestamp" in versions[0]

        store.close()


class TestMultiFounderWorkflow:
    """Test workflow with multi-founder genotypes."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def hs_geno_file(self):
        """Create an 8-founder HS-style file."""
        content = """# HS Rat genotype file
@name:HSRats
@type:hs
Chr\tLocus\tcM\tMb\tHS1\tHS2\tHS3
1\trs001\t1.0\t5.0\tA\tB\tC
1\trs002\t2.0\t6.0\tD\tE\tF
1\trs003\t3.0\t7.0\tG\tH\tA
1\trs004\t4.0\t8.0\tH\tA\tB
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.geno', delete=False) as f:
            f.write(content)
            temp_path = f.name

        yield temp_path
        os.unlink(temp_path)

    def test_multi_founder_storage(self, temp_db, hs_geno_file):
        """Should handle multi-founder genotypes."""
        genotype = parse_genotype_file(hs_geno_file)

        # Should have 8 founders
        assert len(genotype.founders) >= 8

        # H and U should get codes after founders
        assert genotype.het_code == len(genotype.founders)
        assert genotype.unk_code == len(genotype.founders) + 1

        # Store and retrieve
        store = MatrixStore(temp_db)
        store.store_initial("hs_dataset", genotype)

        reconstructed = store.get_matrix("hs_dataset")
        np.testing.assert_array_equal(reconstructed.matrix, genotype.matrix)

        store.close()


class TestDeltaEncodingIntegration:
    """Test delta encoding in full workflow."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_sparse_delta_storage(self, temp_db):
        """Should use sparse delta for small changes."""
        store = MatrixStore(temp_db)
        store.FULL_SNAPSHOT_INTERVAL = 10  # Force deltas

        matrix = GenotypeMatrix(
            matrix=np.zeros((100, 50), dtype=np.uint8),
            markers=[f"m{i}" for i in range(100)],
            samples=[f"s{i}" for i in range(50)],
            chromosomes=["1"] * 100,
            cM=[float(i) for i in range(100)],
            Mb=[float(i) for i in range(100)],
            allele_map={"B": 0, "D": 1},
            founders=["B", "D"],
            het_code=2,
            unk_code=3
        )

        store.store_initial("sparse_test", matrix)

        # Make sparse change (1 cell)
        new_matrix = matrix.matrix.copy()
        new_matrix[50, 25] = 1

        v2 = store.store_update("sparse_test", new_matrix, "test", "single cell")

        # Should be sparse delta
        assert v2.storage_type in ["sparse", "delta"]

        # Verify reconstruction
        reconstructed = store.get_matrix("sparse_test", 2)
        assert reconstructed.matrix[50, 25] == 1
        np.testing.assert_array_equal(reconstructed.matrix, new_matrix)

        store.close()

    def test_full_snapshot_interval(self, temp_db):
        """Should create full snapshots at intervals."""
        store = MatrixStore(temp_db)
        store.FULL_SNAPSHOT_INTERVAL = 3

        matrix = GenotypeMatrix(
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

        store.store_initial("interval_test", matrix)

        # v2, v3: delta, v4: full
        for i in range(2, 6):
            new_matrix = np.full((10, 10), i, dtype=np.uint8)
            v = store.store_update("interval_test", new_matrix, "test", f"v{i}")

            if i % 3 == 1:  # v4
                assert v.storage_type == "full"
            else:
                assert v.storage_type in ["sparse", "delta", "xor"]

        # Verify all reconstruct correctly
        for i in range(1, 6):
            recon = store.get_matrix("interval_test", i)
            if i == 1:
                assert recon.matrix[0, 0] == 0  # v1 is zeros
            else:
                assert recon.matrix[0, 0] == i  # v2 has 2s, v3 has 3s, etc.

        store.close()


class TestCLIWorkflow:
    """Test CLI-like workflow: parse .geno -> pass full GenotypeMatrix -> verify metadata preserved."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def geno_v1_file(self):
        """Create initial genotype file."""
        content = """# File name: test.geno
@name:TestDataset
@type:riset
@mat:B
@pat:D
@het:H
@unk:U
Chr\tLocus\tcM\tMb\tS1\tS2\tS3
1\trs001\t1.0\t5.0\tB\tB\tD
1\trs002\t2.0\t6.0\tD\tH\tU
2\trs003\t0.0\t10.0\tH\tU\tB
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.geno', delete=False) as f:
            f.write(content)
            temp_path = f.name
        yield temp_path
        os.unlink(temp_path)

    @pytest.fixture
    def geno_v2_file(self):
        """Create updated genotype file with different metadata."""
        content = """# File name: test_v2.geno
@name:UpdatedDataset
@type:riset
@mat:A
@pat:T
@het:H
@unk:U
Chr\tLocus\tcM\tMb\tS1\tS2\tS3\tS4
1\trs001\t10.0\t50.0\tA\tA\tT\tA
1\trs002\t20.0\t60.0\tT\tH\tU\tT
1\trs003\t30.0\t70.0\tH\tU\tA\tH
2\trs004\t40.0\t80.0\tA\tT\tT\tH
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.geno', delete=False) as f:
            f.write(content)
            temp_path = f.name
        yield temp_path
        os.unlink(temp_path)

    def test_cli_update_preserves_metadata_on_full_snapshot(self, temp_db, geno_v1_file, geno_v2_file):
        """Simulate CLI: parse .geno, pass full GenotypeMatrix, verify metadata on full snapshot."""
        store = MatrixStore(temp_db)
        store.FULL_SNAPSHOT_INTERVAL = 1  # Force full snapshot

        # v1: Initial import
        genotype_v1 = parse_genotype_file(geno_v1_file)
        store.store_initial("test_dataset", genotype_v1)

        # v2: Update with new .geno file — pass full GenotypeMatrix (like CLI now does)
        genotype_v2 = parse_genotype_file(geno_v2_file)
        v2 = store.store_update("test_dataset", genotype_v2, "cli_user", "Updated with new markers")

        assert v2.storage_type == "full"

        # Reconstruct v2 — metadata should come from v2, not v1
        reconstructed = store.get_matrix("test_dataset", 2)
        assert reconstructed.matrix.shape == (4, 4)  # 4 markers, 4 samples
        assert reconstructed.markers == ["rs001", "rs002", "rs003", "rs004"]
        assert reconstructed.samples == ["S1", "S2", "S3", "S4"]
        assert reconstructed.chromosomes == ["1", "1", "1", "2"]
        assert reconstructed.cM == [10.0, 20.0, 30.0, 40.0]
        assert reconstructed.Mb == [50.0, 60.0, 70.0, 80.0]
        assert reconstructed.allele_map == {"A": 0, "T": 1}
        assert reconstructed.founders == ["A", "T"]
        assert reconstructed.cross_type == "riset"
        assert reconstructed.dataset_name == "UpdatedDataset"

        store.close()

    def test_cli_update_metadata_on_delta(self, temp_db, geno_v1_file):
        """Simulate CLI: delta update should reuse metadata from previous full snapshot."""
        store = MatrixStore(temp_db)
        store.FULL_SNAPSHOT_INTERVAL = 10  # Force delta

        # v1: Initial import
        genotype_v1 = parse_genotype_file(geno_v1_file)
        store.store_initial("test_dataset", genotype_v1)

        # v2: Small change — pass full GenotypeMatrix, but delta will be used
        genotype_v2 = parse_genotype_file(geno_v1_file)
        genotype_v2.matrix[0, 0] = 1  # Small change

        v2 = store.store_update("test_dataset", genotype_v2, "cli_user", "Small correction")
        assert v2.storage_type == "delta"

        # Reconstruct v2 — metadata should come from v1
        reconstructed = store.get_matrix("test_dataset", 2)
        assert reconstructed.markers == genotype_v1.markers
        assert reconstructed.samples == genotype_v1.samples
        assert reconstructed.allele_map == genotype_v1.allele_map
        assert reconstructed.founders == genotype_v1.founders

        store.close()

    def test_cli_update_with_shape_change(self, temp_db, geno_v1_file, geno_v2_file):
        """Simulate CLI: shape change forces full snapshot, metadata from new .geno preserved."""
        store = MatrixStore(temp_db)

        # v1: Initial import (3x3)
        genotype_v1 = parse_genotype_file(geno_v1_file)
        store.store_initial("test_dataset", genotype_v1)

        # v2: Different shape (4x4) — forces full snapshot
        genotype_v2 = parse_genotype_file(geno_v2_file)
        v2 = store.store_update("test_dataset", genotype_v2, "cli_user", "Added marker and sample")

        assert v2.storage_type == "full"

        # Reconstruct v2 — metadata from new .geno
        reconstructed = store.get_matrix("test_dataset", 2)
        assert reconstructed.matrix.shape == (4, 4)
        assert reconstructed.markers == ["rs001", "rs002", "rs003", "rs004"]
        assert reconstructed.samples == ["S1", "S2", "S3", "S4"]
        assert reconstructed.cross_type == "riset"

        # Verify hash chain intact
        valid, errors = store.verify_dataset("test_dataset")
        assert valid is True
        assert len(errors) == 0

        store.close()


class TestErrorRecovery:
    """Test error handling and recovery."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_corruption_detection(self, temp_db):
        """Should detect corrupted hash chain."""
        genotype = GenotypeMatrix(
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

        store = MatrixStore(temp_db)
        store.store_initial("corrupt_test", genotype)

        # Verify should pass
        valid, errors = store.verify_dataset("corrupt_test")
        assert valid is True

        store.close()
