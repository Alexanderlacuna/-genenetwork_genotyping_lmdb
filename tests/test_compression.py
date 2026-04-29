"""Tests for payload compression.

Compression is a transparent storage-layer concern.
The delta encoder and hash chain are agnostic to it.
"""

import pytest
import numpy as np
import tempfile
import shutil

from geno_storage.matrix_store import MatrixStore
from geno_storage.models import GenotypeMatrix
from geno_storage.compression import compress_payload, decompress_payload, get_compressor


class TestCompressionPlugin:
    """Test compression plugin system."""

    def test_null_compressor(self):
        """Null compressor passes bytes through unchanged."""
        data = b"hello world"
        compressed = compress_payload(data, None)
        assert compressed == data
        assert decompress_payload(compressed, None) == data

    def test_none_compressor(self):
        """'none' is alias for null compressor."""
        data = b"hello world"
        assert compress_payload(data, "none") == data
        assert decompress_payload(data, "none") == data

    def test_zlib_compressor(self):
        """zlib compressor actually reduces size for repetitive data."""
        data = b"A" * 1000
        compressed = compress_payload(data, "zlib")
        assert len(compressed) < len(data)
        assert decompress_payload(compressed, "zlib") == data

    def test_zlib_roundtrip(self):
        """zlib compress + decompress is identity."""
        data = bytes(range(256)) * 10
        compressed = compress_payload(data, "zlib")
        decompressed = decompress_payload(compressed, "zlib")
        assert decompressed == data

    def test_unknown_compressor_raises(self):
        """Unknown compressor name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown compression algorithm"):
            get_compressor("bogus")


class TestCompressedStore:
    """Test MatrixStore with compression enabled."""

    @pytest.fixture
    def temp_db(self):
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)

    @pytest.fixture
    def sample_matrix(self):
        return GenotypeMatrix(
            matrix=np.array([[0, 1, 2], [1, 0, 3]], dtype=np.uint8),
            markers=["m1", "m2"],
            samples=["s1", "s2", "s3"],
            chromosomes=["1", "1"],
            cM=[1.0, 2.0],
            Mb=[5.0, 6.0],
            allele_map={"B": 0, "D": 1},
            founders=["B", "D"],
            het_code=2,
            unk_code=3,
            dataset_name="TestDS"
        )

    def test_store_initial_with_compression(self, temp_db, sample_matrix):
        """Store initial with compression stores compressed payload."""
        store = MatrixStore(temp_db)
        version = store.store_initial("test", sample_matrix, compression="zlib")

        assert version.compression == "zlib"
        assert version.storage_type == "full"

        # For tiny matrices, compressed may be larger due to zlib overhead.
        # Just verify it's different from raw and round-trips correctly.
        raw = store.delta_encoder.encode_full(sample_matrix.matrix)
        assert version.payload != raw  # compressed is different

        store.close()

    def test_store_initial_without_compression(self, temp_db, sample_matrix):
        """Store initial without compression stores raw payload."""
        store = MatrixStore(temp_db)
        version = store.store_initial("test", sample_matrix)

        assert version.compression is None

        # Payload should be raw bytes
        raw = store.delta_encoder.encode_full(sample_matrix.matrix)
        assert version.payload == raw

        store.close()

    def test_large_matrix_compression_saves_space(self, temp_db):
        """Large repetitive matrices compress significantly."""
        large = GenotypeMatrix(
            matrix=np.zeros((1000, 500), dtype=np.uint8),
            markers=[f"m{i}" for i in range(1000)],
            samples=[f"s{i}" for i in range(500)],
            chromosomes=["1"] * 1000,
            cM=[float(i) for i in range(1000)],
            Mb=[float(i) for i in range(1000)],
            allele_map={"B": 0, "D": 1},
            founders=["B", "D"],
            het_code=2,
            unk_code=3,
            dataset_name="LargeDS"
        )
        store = MatrixStore(temp_db)
        version = store.store_initial("large", large, compression="zlib")

        raw = store.delta_encoder.encode_full(large.matrix)
        assert len(version.payload) < len(raw) * 0.5  # at least 50% smaller

        store.close()

    def test_reconstruct_compressed(self, temp_db, sample_matrix):
        """Reconstruct works with compressed payloads."""
        store = MatrixStore(temp_db)
        store.store_initial("test", sample_matrix, compression="zlib")

        reconstructed = store.get_matrix("test", 1)
        np.testing.assert_array_equal(reconstructed.matrix, sample_matrix.matrix)
        assert reconstructed.markers == sample_matrix.markers
        assert reconstructed.samples == sample_matrix.samples

        store.close()

    def test_verify_compressed(self, temp_db, sample_matrix):
        """Verification passes on compressed store."""
        store = MatrixStore(temp_db)
        store.store_initial("test", sample_matrix, compression="zlib")

        # Add a delta
        new_matrix = sample_matrix.matrix.copy()
        new_matrix[0, 0] = 1
        store.store_update("test", new_matrix, "user", "fix")

        valid, errors = store.verify_dataset("test")
        assert valid is True
        assert len(errors) == 0

        store.close()

    def test_update_inherits_compression(self, temp_db, sample_matrix):
        """Updates use the store's compression config automatically."""
        store = MatrixStore(temp_db)
        store.store_initial("test", sample_matrix, compression="zlib")

        # Update without specifying compression
        new_matrix = sample_matrix.matrix.copy()
        new_matrix[0, 0] = 1
        v2 = store.store_update("test", new_matrix, "user", "fix")

        # Should inherit zlib from store config
        assert v2.compression == "zlib"

        store.close()

    def test_config_persisted_in_lmdb(self, temp_db, sample_matrix):
        """Compression config is read from LMDB on re-open."""
        store1 = MatrixStore(temp_db)
        store1.store_initial("test", sample_matrix, compression="zlib")
        store1.close()

        # Re-open store — should read compression from LMDB config
        store2 = MatrixStore(temp_db)
        with store2.env.begin() as txn:
            config = store2.get_store_config(txn, 'compression')
        assert config == "zlib"

        # Update should use zlib automatically
        new_matrix = sample_matrix.matrix.copy()
        new_matrix[0, 0] = 1
        v2 = store2.store_update("test", new_matrix, "user", "fix")
        assert v2.compression == "zlib"

        store2.close()

    def test_config_read_on_reopen(self, temp_db, sample_matrix):
        """Compression config is read from LMDB on every open."""
        store = MatrixStore(temp_db)
        store.store_initial("test", sample_matrix, compression="zlib")
        store.close()

        # Re-open without specifying compression — reads from LMDB
        store2 = MatrixStore(temp_db)
        with store2.env.begin() as txn:
            config = store2.get_store_config(txn, 'compression')
        assert config == "zlib"

        store2.close()

    def test_mixed_compression_rejected(self, temp_db, sample_matrix):
        """Adding a second dataset with different compression is rejected."""
        store = MatrixStore(temp_db)
        store.store_initial("ds1", sample_matrix, compression="zlib")

        # Trying to pass an explicit different compression raises
        with pytest.raises(ValueError, match="Mixed compression"):
            store.store_initial("ds2", sample_matrix, compression="lz4")

        store.close()

    def test_uncompressed_dataset_after_compressed_rejected(self, temp_db, sample_matrix):
        """Passing 'none' as explicit compression on a compressed store is rejected."""
        store = MatrixStore(temp_db)
        store.store_initial("ds1", sample_matrix, compression="zlib")

        with pytest.raises(ValueError, match="Mixed compression"):
            store.store_initial("ds2", sample_matrix, compression="none")

        store.close()
