"""
Tests for .geno file parser with multi-founder support.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import os

from geno_storage.geno_parser import parse_genotype_file
from geno_storage.models import GenotypeMatrix
from geno_storage.matrix_ops import get_allele_frequencies


class TestBXDStyleParsing:
    """Test parsing of BXD-style 2-founder files."""
    
    @pytest.fixture
    def bxd_geno_file(self):
        """Create a temporary BXD-style genotype file."""
        content = """# File name: test.geno
# Description: Test genotype file
@name:BXD
@type:riset
@mat:B
@pat:D
@het:H
@unk:U
Chr\tLocus\tcM\tMb\tBXD1\tBXD2\tBXD5
1\trs123\t1.5\t3.0\tB\tB\tD
1\trs456\t2.5\t4.0\tD\tH\tU
2\trs789\t0.0\t5.0\tH\tU\tB
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.geno', delete=False) as f:
            f.write(content)
            temp_path = f.name
        
        yield temp_path
        os.unlink(temp_path)
    
    def test_parse_bxd(self, bxd_geno_file):
        """Should parse BXD file correctly."""
        result = parse_genotype_file(bxd_geno_file)
        
        assert isinstance(result, GenotypeMatrix)
        assert result.dataset_name == "BXD"
        assert result.cross_type == "riset"
    
    def test_matrix_shape(self, bxd_geno_file):
        """Should have correct dimensions."""
        result = parse_genotype_file(bxd_geno_file)
        
        # 3 markers, 3 samples
        assert result.matrix.shape == (3, 3)
    
    def test_sample_names(self, bxd_geno_file):
        """Should extract sample names."""
        result = parse_genotype_file(bxd_geno_file)
        
        assert result.samples == ["BXD1", "BXD2", "BXD5"]
    
    def test_marker_names(self, bxd_geno_file):
        """Should extract marker names."""
        result = parse_genotype_file(bxd_geno_file)
        
        assert result.markers == ["rs123", "rs456", "rs789"]
    
    def test_chromosomes(self, bxd_geno_file):
        """Should extract chromosome info."""
        result = parse_genotype_file(bxd_geno_file)
        
        assert result.chromosomes == ["1", "1", "2"]
    
    def test_positions(self, bxd_geno_file):
        """Should extract cM and Mb position info."""
        result = parse_genotype_file(bxd_geno_file)
        
        assert result.cM == [1.5, 2.5, 0.0]
        assert result.Mb == [3.0, 4.0, 5.0]
    
    def test_allele_encoding(self, bxd_geno_file):
        """Should encode alleles correctly."""
        result = parse_genotype_file(bxd_geno_file)
        
        # B=0, D=1, H=2, U=3
        expected = np.array([
            [0, 0, 1],  # B, B, D
            [1, 2, 3],  # D, H, U
            [2, 3, 0],  # H, U, B
        ], dtype=np.uint8)
        
        np.testing.assert_array_equal(result.matrix, expected)
    
    def test_founders(self, bxd_geno_file):
        """Should identify founders."""
        result = parse_genotype_file(bxd_geno_file)
        
        assert result.founders == ["B", "D"]
        assert result.mat_allele == "B"
        assert result.pat_allele == "D"
    
    def test_allele_map(self, bxd_geno_file):
        """Should create correct allele map."""
        result = parse_genotype_file(bxd_geno_file)
        
        assert result.allele_map == {"B": 0, "D": 1}
        assert result.het_code == 2
        assert result.unk_code == 3


class TestNumericGenotypes:
    """Test parsing of files with numeric genotypes."""
    
    @pytest.fixture
    def numeric_geno_file(self):
        """Create a file with numeric genotypes."""
        content = """@name:NumericTest
@type:riset
Chr\tLocus\tcM\tMb\tS1\tS2\tS3
1\trs1\t1.0\t5.0\t0\t1\t2
1\trs2\t2.0\t6.0\t1\t2\t3
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.geno', delete=False) as f:
            f.write(content)
            temp_path = f.name
        
        yield temp_path
        os.unlink(temp_path)
    
    def test_numeric_parsing(self, numeric_geno_file):
        """Should parse numeric genotypes."""
        result = parse_genotype_file(numeric_geno_file)
        
        expected = np.array([
            [0, 1, 2],
            [1, 2, 3],
        ], dtype=np.uint8)
        
        np.testing.assert_array_equal(result.matrix, expected)


class TestMultiFounder:
    """Test parsing of multi-founder crosses (HS, DO)."""
    
    @pytest.fixture
    def hs_geno_file(self):
        """Create an HS-style 8-founder file."""
        content = """@name:HSTest
@type:hs
Chr\tLocus\tcM\tMb\tS1\tS2\tS3
1\trs1\t1.0\t5.0\tA\tB\tC
1\trs2\t2.0\t6.0\tD\tE\tH
1\trs3\t3.0\t7.0\tF\tG\tA
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.geno', delete=False) as f:
            f.write(content)
            temp_path = f.name
        
        yield temp_path
        os.unlink(temp_path)
    
    def test_multi_founder_encoding(self, hs_geno_file):
        """Should handle 8 founders."""
        result = parse_genotype_file(hs_geno_file)
        
        # A=0, B=1, C=2, D=3, E=4, F=5, G=6, H=7
        # H=8 (het), U=9 (unknown)
        assert len(result.founders) >= 8
        assert result.het_code == len(result.founders)
    
    def test_multi_founder_values(self, hs_geno_file):
        """Should encode multi-founder values correctly."""
        result = parse_genotype_file(hs_geno_file)
        
        # Check first row: A, B, C -> 0, 1, 2
        assert result.matrix[0, 0] == 0  # A
        assert result.matrix[0, 1] == 1  # B
        assert result.matrix[0, 2] == 2  # C


class TestF2Cross:
    """Test F2 intercross parsing."""
    
    @pytest.fixture
    def f2_geno_file(self):
        """Create an F2 cross file."""
        content = """@name:F2Test
@type:f2
@mat:A
@pat:B
Chr\tLocus\tcM\tMb\tS1\tS2\tS3
1\trs1\t1.0\t5.0\tAA\tAB\tBB
1\trs2\t2.0\t6.0\tAB\tBB\tAA
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.geno', delete=False) as f:
            f.write(content)
            temp_path = f.name
        
        yield temp_path
        os.unlink(temp_path)
    
    def test_f2_encoding(self, f2_geno_file):
        """Should parse F2 genotypes."""
        result = parse_genotype_file(f2_geno_file)
        
        # AA=0, AB=1, BB=2
        expected = np.array([
            [0, 1, 2],
            [1, 2, 0],
        ], dtype=np.uint8)
        
        np.testing.assert_array_equal(result.matrix, expected)


class TestMetadataExtraction:
    """Test metadata extraction."""
    
    @pytest.fixture
    def metadata_file(self):
        """Create file with extensive metadata."""
        content = """# File name: test.geno
# Citation: Smith et al. 2020
# Description: Test file
@name:TestSet
@type:riset
@mat:B
@pat:D
Chr\tLocus\tcM\tMb\tS1
1\trs1\t1.0\t5.0\tB
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.geno', delete=False) as f:
            f.write(content)
            temp_path = f.name
        
        yield temp_path
        os.unlink(temp_path)
    
    def test_file_metadata(self, metadata_file):
        """Should extract file metadata."""
        result = parse_genotype_file(metadata_file)
        
        assert "File name" in result.file_metadata
        assert "Citation" in result.file_metadata
    
    def test_dataset_metadata(self, metadata_file):
        """Should extract dataset metadata."""
        result = parse_genotype_file(metadata_file)
        
        assert result.dataset_name == "TestSet"


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def empty_file(self):
        """Create empty genotype file."""
        content = """@name:Empty
@type:riset
Chr\tLocus\tcM\tMb
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.geno', delete=False) as f:
            f.write(content)
            temp_path = f.name
        
        yield temp_path
        os.unlink(temp_path)
    
    @pytest.fixture
    def missing_values_file(self):
        """Create file with missing values."""
        content = """@name:Missing
@type:riset
Chr\tLocus\tcM\tMb\tS1\tS2
1\trs1\t1.0\t5.0\tB\t-
1\trs2\t2.0\t6.0\t-\tD
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.geno', delete=False) as f:
            f.write(content)
            temp_path = f.name
        
        yield temp_path
        os.unlink(temp_path)
    
    def test_missing_values(self, missing_values_file):
        """Should handle missing values (-)."""
        result = parse_genotype_file(missing_values_file)

        # '-' should become unknown (U/3)
        assert result.matrix[0, 1] == result.unk_code
        assert result.matrix[1, 0] == result.unk_code


class TestCommentHandling:
    """Test parsing of various comment styles."""

    @pytest.fixture
    def comment_without_colon(self):
        """Create file with # comment lacking colon."""
        content = """# HS Rat Genotype File
# Description: 8-founder Heterogeneous Stock rat cross
# Source: http://www.genenetwork.org
@name:HSRats1
@type:hs
@mat:A
@pat:H
@het:H
@unk:U
Chr\tLocus\tcM\tMb\tHS1\tHS2\tHS3\tHS4\tHS5\tHS6\tHS7\tHS8
1\trs001\t1.0\t5.0\tA\tB\tC\tD\tA\tE\tF\tG
1\trs002\t2.0\t6.0\tD\tE\tF\tG\tH\tA\tB\tC
1\trs003\t3.0\t7.0\tH\tA\tB\tC\tD\tE\tF\tG
2\trs004\t0.5\t12.0\tB\tC\tD\tE\tF\tG\tH\tA
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.geno', delete=False) as f:
            f.write(content)
            temp_path = f.name
        yield temp_path
        os.unlink(temp_path)

    @pytest.fixture
    def mixed_comments(self):
        """Create file with mixed comment styles."""
        content = """# File name: test.geno
# This is a plain comment without colon
# Another plain comment
# Description: Test file with mixed comments
@name:MixedTest
@type:riset
@mat:B
@pat:D
Chr\tLocus\tcM\tMb\tS1\tS2\tS3
1\trs1\t1.0\t5.0\tB\tD\tH
1\trs2\t2.0\t6.0\tD\tH\tU
2\trs3\t0.0\t10.0\tH\tB\tD
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.geno', delete=False) as f:
            f.write(content)
            temp_path = f.name
        yield temp_path
        os.unlink(temp_path)

    @pytest.fixture
    def many_plain_comments(self):
        """Create file with many plain comments before header."""
        content = """# File entered in GN database on Dec 19, 2018 by Arthur Centeno
# Coordinates of Markers and Assembly
# Genotypes information line without colon
# Material and Cases description
# Breeding information
# Errors and Corrections note
# Funding information
@name:ManyComments
@type:riset
@mat:B
@pat:D
@het:H
@unk:U
Chr\tLocus\tcM\tMb\tS1\tS2
1\trs1\t1.0\t5.0\tB\tD
1\trs2\t2.0\t6.0\tD\tH
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.geno', delete=False) as f:
            f.write(content)
            temp_path = f.name
        yield temp_path
        os.unlink(temp_path)

    def test_comment_without_colon(self, comment_without_colon):
        """Should skip # comments without colon, not treat as header."""
        result = parse_genotype_file(comment_without_colon)

        # Should NOT have picked up "# HS Rat Genotype File" as samples
        assert result.samples == ["HS1", "HS2", "HS3", "HS4", "HS5", "HS6", "HS7", "HS8"]
        assert result.markers == ["rs001", "rs002", "rs003", "rs004"]
        assert result.matrix.shape == (4, 8)
        assert result.cross_type == "hs"
        assert result.dataset_name == "HSRats1"

    def test_mixed_comments(self, mixed_comments):
        """Should handle mix of key:value and plain comments."""
        result = parse_genotype_file(mixed_comments)

        assert result.samples == ["S1", "S2", "S3"]
        assert result.markers == ["rs1", "rs2", "rs3"]
        assert result.matrix.shape == (3, 3)
        assert result.cross_type == "riset"
        # Should have parsed key:value comments
        assert "File name" in result.file_metadata
        assert "Description" in result.file_metadata

    def test_many_plain_comments(self, many_plain_comments):
        """Should skip many consecutive plain comments."""
        result = parse_genotype_file(many_plain_comments)

        assert result.samples == ["S1", "S2"]
        assert result.markers == ["rs1", "rs2"]
        assert result.matrix.shape == (2, 2)
        assert result.cross_type == "riset"
        assert result.dataset_name == "ManyComments"

    def test_real_bxd_file(self):
        """Should parse real BXD.geno file correctly."""
        bxd_path = Path('/home/kabui/genotype_files/genotype/BXD.geno')
        if not bxd_path.exists():
            pytest.skip("BXD.geno not found")

        result = parse_genotype_file(bxd_path)

        # Should have ~7320 markers and ~198 samples
        assert result.matrix.shape[0] > 7000
        assert result.matrix.shape[1] > 100
        assert result.samples[0] == "BXD1"
        assert result.founders == ["B", "D"]
        assert result.cross_type == "riset"

    def test_real_hsrat_file(self):
        """Should parse real HSRats1.geno file correctly."""
        hs_path = Path('/home/kabui/genotype_files/genotype/HSRats1.geno')
        if not hs_path.exists():
            pytest.skip("HSRats1.geno not found")

        result = parse_genotype_file(hs_path)

        # Should have ~11 markers and 8 samples
        assert result.matrix.shape == (11, 8)
        assert result.samples == ["HS1", "HS2", "HS3", "HS4", "HS5", "HS6", "HS7", "HS8"]
        assert len(result.founders) >= 8
        assert result.cross_type == "hs"


class TestGenotypeMatrixMethods:
    """Test GenotypeMatrix utility methods."""
    
    def test_get_allele_frequencies(self):
        """Test allele frequency calculation."""
        matrix = GenotypeMatrix(
            matrix=np.array([[0, 1], [1, 2]], dtype=np.uint8),
            markers=["m1", "m2"],
            samples=["s1", "s2"],
            chromosomes=["1", "1"],
            cM=[1.0, 2.0],
            Mb=[5.0, 6.0],
            allele_map={"A": 0, "B": 1},
            founders=["A", "B"],
            het_code=2,
            unk_code=3
        )
        
        freqs = get_allele_frequencies(matrix)
        assert freqs[0] == 1  # One 'A'
        assert freqs[1] == 2  # Two 'B's
        assert freqs[2] == 1  # One 'H'
