"""Tests for .geno export (round-trip)."""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path

from geno_storage.geno_parser import parse_genotype_file, GenotypeMatrix
from geno_storage.geno_writer import export_genotype_file
from geno_storage.matrix_store import MatrixStore


class TestExportBasic:
    """Test basic export functionality."""

    @pytest.fixture
    def temp_dir(self):
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
            dataset_name="TestDS",
            cross_type="riset",
            mat_allele="B",
            pat_allele="D"
        )

    def test_export_creates_file(self, temp_dir, sample_matrix):
        """Export should create a file."""
        out_path = Path(temp_dir) / "test.geno"
        result = export_genotype_file(sample_matrix, out_path)
        assert result.exists()
        assert result.stat().st_size > 0

    def test_export_no_comments_by_default(self, temp_dir, sample_matrix):
        """Export should NOT include comments by default."""
        out_path = Path(temp_dir) / "test.geno"
        export_genotype_file(sample_matrix, out_path)
        content = out_path.read_text()
        assert "#" not in content
        assert "@name:TestDS" in content

    def test_export_with_comments(self, temp_dir, sample_matrix):
        """Export with include_comments=True should include comments."""
        out_path = Path(temp_dir) / "test.geno"
        export_genotype_file(sample_matrix, out_path, include_comments=True)
        content = out_path.read_text()
        assert "# Exported" in content
        assert "TestDS" in content

    def test_export_has_metadata(self, temp_dir, sample_matrix):
        """Export should include @metadata lines."""
        out_path = Path(temp_dir) / "test.geno"
        export_genotype_file(sample_matrix, out_path)
        content = out_path.read_text()
        assert "@name:TestDS" in content
        assert "@type:riset" in content
        assert "@mat:B" in content
        assert "@pat:D" in content
        assert "@het:H" in content
        assert "@unk:U" in content

    def test_export_has_header_row(self, temp_dir, sample_matrix):
        """Export should include tab-separated header with samples."""
        out_path = Path(temp_dir) / "test.geno"
        export_genotype_file(sample_matrix, out_path)
        lines = out_path.read_text().strip().split("\n")
        # Find header line (first non-comment, non-metadata line)
        header_line = None
        for line in lines:
            if not line.startswith("#") and not line.startswith("@"):
                header_line = line
                break
        assert header_line is not None
        parts = header_line.split("\t")
        assert parts[0] == "Chr"
        assert parts[1] == "Locus"
        assert parts[2] == "cM"
        assert parts[3] == "Mb"
        assert parts[4] == "s1"
        assert parts[5] == "s2"
        assert parts[6] == "s3"

    def test_export_has_data_rows(self, temp_dir, sample_matrix):
        """Export should include data rows with correct symbols."""
        out_path = Path(temp_dir) / "test.geno"
        export_genotype_file(sample_matrix, out_path)
        lines = out_path.read_text().strip().split("\n")
        data_lines = [l for l in lines if not l.startswith("#") and not l.startswith("@") and l.startswith("1")]
        assert len(data_lines) == 2
        # First row: [0, 1, 2] -> B, D, H
        parts = data_lines[0].split("\t")
        assert parts[0] == "1"
        assert parts[1] == "m1"
        assert parts[4] == "B"
        assert parts[5] == "D"
        assert parts[6] == "H"
        # Second row: [1, 0, 3] -> D, B, U
        parts = data_lines[1].split("\t")
        assert parts[4] == "D"
        assert parts[5] == "B"
        assert parts[6] == "U"


class TestRoundTrip:
    """Test round-trip: parse -> store -> reconstruct -> export -> re-parse."""

    @pytest.fixture
    def temp_db(self):
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)

    def test_round_trip_synthetic(self, temp_db):
        """Round-trip a synthetic GenotypeMatrix through LMDB and back."""
        store = MatrixStore(temp_db)

        original = GenotypeMatrix(
            matrix=np.array([
                [0, 1, 2, 3],
                [1, 0, 3, 2],
                [2, 3, 0, 1],
            ], dtype=np.uint8),
            markers=["mk1", "mk2", "mk3"],
            samples=["A", "B", "C", "D"],
            chromosomes=["1", "2", "X"],
            cM=[1.5, 2.5, 3.5],
            Mb=[5.5, 6.5, 7.5],
            allele_map={"B": 0, "D": 1},
            founders=["B", "D"],
            het_code=2,
            unk_code=3,
            dataset_name="RoundTrip",
            cross_type="riset",
            mat_allele="B",
            pat_allele="D"
        )

        store.store_initial("RoundTrip", original)
        reconstructed = store.get_matrix("RoundTrip", 1)

        out_path = Path(temp_db) / "exported.geno"
        export_genotype_file(reconstructed, out_path)

        reparsed = parse_genotype_file(out_path)

        assert np.array_equal(original.matrix, reparsed.matrix)
        assert original.markers == reparsed.markers
        assert original.samples == reparsed.samples
        assert original.chromosomes == reparsed.chromosomes
        assert np.allclose(original.cM, reparsed.cM)
        assert np.allclose(original.Mb, reparsed.Mb)
        assert original.allele_map == reparsed.allele_map
        assert original.founders == reparsed.founders
        assert original.dataset_name == reparsed.dataset_name
        assert original.cross_type == reparsed.cross_type

        store.close()

    def test_round_trip_bxd_file(self, temp_db):
        """Round-trip a real BXD.geno file through LMDB and back."""
        store = MatrixStore(temp_db)

        original = parse_genotype_file("/home/kabui/genotype_files/genotype/BXD.geno")
        store.store_initial("BXD", original)
        reconstructed = store.get_matrix("BXD", 1)

        out_path = Path(temp_db) / "BXD_roundtrip.geno"
        export_genotype_file(reconstructed, out_path)

        reparsed = parse_genotype_file(out_path)

        assert np.array_equal(original.matrix, reparsed.matrix)
        assert original.markers == reparsed.markers
        assert original.samples == reparsed.samples
        assert original.chromosomes == reparsed.chromosomes
        assert np.allclose(original.cM, reparsed.cM)
        assert np.allclose(original.Mb, reparsed.Mb)
        assert original.allele_map == reparsed.allele_map
        assert original.founders == reparsed.founders
        assert original.dataset_name == reparsed.dataset_name
        assert original.cross_type == reparsed.cross_type

        store.close()

    def test_round_trip_hsrat_file(self, temp_db):
        """Round-trip a real HSRats1.geno file through LMDB and back."""
        store = MatrixStore(temp_db)

        original = parse_genotype_file("/home/kabui/genotype_files/genotype/HSRats1.geno")
        store.store_initial("HSRats", original)
        reconstructed = store.get_matrix("HSRats", 1)

        out_path = Path(temp_db) / "HSRats_roundtrip.geno"
        export_genotype_file(reconstructed, out_path)

        reparsed = parse_genotype_file(out_path)

        assert np.array_equal(original.matrix, reparsed.matrix)
        assert original.markers == reparsed.markers
        assert original.samples == reparsed.samples
        assert original.chromosomes == reparsed.chromosomes
        assert np.allclose(original.cM, reparsed.cM)
        assert np.allclose(original.Mb, reparsed.Mb)
        assert original.allele_map == reparsed.allele_map
        assert original.founders == reparsed.founders
        assert original.dataset_name == reparsed.dataset_name
        assert original.cross_type == reparsed.cross_type

        store.close()

    def test_multi_founder_round_trip(self, temp_db):
        """Round-trip a multi-founder matrix."""
        store = MatrixStore(temp_db)

        original = GenotypeMatrix(
            matrix=np.array([
                [0, 1, 2, 3, 4, 5],
                [5, 4, 3, 2, 1, 0],
            ], dtype=np.uint8),
            markers=["m1", "m2"],
            samples=["s1", "s2", "s3", "s4", "s5", "s6"],
            chromosomes=["1", "1"],
            cM=[1.0, 2.0],
            Mb=[5.0, 6.0],
            allele_map={"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5},
            founders=["A", "B", "C", "D", "E", "F"],
            het_code=6,
            unk_code=7,
            dataset_name="MultiFounder",
            cross_type="hs"
        )

        store.store_initial("MultiFounder", original)
        reconstructed = store.get_matrix("MultiFounder", 1)

        out_path = Path(temp_db) / "multi.geno"
        export_genotype_file(reconstructed, out_path)

        reparsed = parse_genotype_file(out_path)

        assert np.array_equal(original.matrix, reparsed.matrix)
        assert original.markers == reparsed.markers
        assert original.samples == reparsed.samples
        assert original.allele_map == reparsed.allele_map
        assert original.founders == reparsed.founders

        store.close()
