"""Unit tests for extracted pure functions.

These tests verify calculations in isolation — no file I/O, no LMDB.
Following Grokking Simplicity: pure functions are the easiest to test.
"""

import pytest
import numpy as np

from geno_storage.geno_parser import detect_encoding, parse_headers
from geno_storage.matrix_store import encode_metadata, decode_metadata
from geno_storage.reconstruction import reconstruct_from_payloads
from geno_storage.delta import DeltaEncoder
from geno_storage.models import GenotypeMatrix


class TestDetectEncoding:
    """Test encoding detection in isolation."""

    def test_riset_with_mat_pat(self):
        """Should detect 2-founder RI when mat/pat given."""
        metadata = {'type': 'riset', 'mat': 'B', 'pat': 'D'}
        raw_lines = ["1\trs1\t1.0\t5.0\tB\tD\tH"]
        result = detect_encoding(metadata, raw_lines)

        assert result['founders'] == ['B', 'D']
        assert result['allele_map'] == {'B': 0, 'D': 1}
        assert result['het_symbol'] == 'H'
        assert result['unk_symbol'] == 'U'
        assert result['cross_type'] == 'riset'

    def test_riset_default_founders(self):
        """Should default to B/D when mat/pat missing."""
        metadata = {'type': 'riset'}
        raw_lines = ["1\trs1\t1.0\t5.0\tB\tD\tH"]
        result = detect_encoding(metadata, raw_lines)

        assert result['founders'] == ['B', 'D']
        assert result['allele_map'] == {'B': 0, 'D': 1}

    def test_f2_intercross(self):
        """Should detect F2 encoding."""
        metadata = {'type': 'f2', 'mat': 'A', 'pat': 'B'}
        raw_lines = ["1\trs1\t1.0\t5.0\tAA\tAB\tBB"]
        result = detect_encoding(metadata, raw_lines)

        assert result['founders'] == ['A', 'B']
        assert result['allele_map']['AA'] == 0
        assert result['allele_map']['AB'] == 1
        assert result['allele_map']['BB'] == 2
        assert result['cross_type'] == 'f2'

    def test_multi_founder_hs(self):
        """Should detect 8-founder HS."""
        metadata = {'type': 'hs'}
        raw_lines = ["1\trs1\t1.0\t5.0\tA\tB\tC\tD\tE\tF\tG\tH"]
        result = detect_encoding(metadata, raw_lines)

        assert len(result['founders']) == 8
        assert result['allele_map']['A'] == 0
        assert result['allele_map']['H'] == 7
        assert result['het_symbol'] == 'H'
        assert result['cross_type'] == 'hs'

    def test_numeric_genotypes(self):
        """Should detect numeric encoding."""
        metadata = {'type': 'riset'}
        raw_lines = ["1\trs1\t1.0\t5.0\t0\t1\t2\t3"]
        result = detect_encoding(metadata, raw_lines)

        assert result['allele_map'] == {'0': 0, '1': 1, '2': 2, '3': 3}
        assert result['founders'] == ['0', '1']
        assert result['het_symbol'] == '2'
        assert result['unk_symbol'] == '3'

    def test_auto_detect_from_alleles(self):
        """Should auto-detect from sample alleles when type unknown."""
        metadata = {}
        raw_lines = ["1\trs1\t1.0\t5.0\tA\tT\tA"]
        result = detect_encoding(metadata, raw_lines)

        assert result['founders'] == ['A', 'T']
        assert result['allele_map'] == {'A': 0, 'T': 1}
        assert result['cross_type'] == 'unknown'

    def test_empty_raw_lines(self):
        """Should handle empty raw lines with defaults."""
        metadata = {'type': 'riset'}
        raw_lines = []
        result = detect_encoding(metadata, raw_lines)

        assert result['founders'] == ['B', 'D']
        assert result['cross_type'] == 'riset'


class TestParseHeaders:
    """Test header parsing in isolation."""

    def test_parse_headers_simple(self):
        """Should parse metadata, comments, and header row."""
        from io import StringIO
        content = """# File name: test.geno
@name:TestSet
@type:riset
Chr\tLocus\tcM\tMb\tS1\tS2
1\trs1\t1.0\t5.0\tB\tD
"""
        f = StringIO(content)
        result = parse_headers(f)

        assert result.metadata == {'name': 'TestSet', 'type': 'riset'}
        assert 'File name' in result.file_info
        assert result.header == ['Chr', 'Locus', 'cM', 'Mb', 'S1', 'S2']
        assert result.raw_lines == ['1\trs1\t1.0\t5.0\tB\tD']

    def test_parse_headers_plain_comments(self):
        """Should skip plain comments without colon."""
        from io import StringIO
        content = """# Plain comment without colon
@name:Test
Chr\tLocus\tcM\tMb\tS1
1\trs1\t1.0\t5.0\tB
"""
        f = StringIO(content)
        result = parse_headers(f)

        assert result.file_info == {}
        assert result.metadata == {'name': 'Test'}
        assert result.raw_lines == ['1\trs1\t1.0\t5.0\tB']


class TestMetadataRoundTrip:
    """Test encode/decode metadata in isolation."""

    @pytest.fixture
    def sample_matrix(self):
        return GenotypeMatrix(
            matrix=np.array([[0, 1], [1, 0]], dtype=np.uint8),
            markers=["m1", "m2"],
            samples=["s1", "s2"],
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
            pat_allele="D",
            file_metadata={"citation": "Smith 2020"}
        )

    def test_encode_decode_roundtrip(self, sample_matrix):
        """Metadata should survive encode/decode round-trip."""
        encoded = encode_metadata(sample_matrix)
        decoded = decode_metadata(encoded)

        assert decoded['markers'] == ["m1", "m2"]
        assert decoded['samples'] == ["s1", "s2"]
        assert decoded['chromosomes'] == ["1", "1"]
        assert decoded['cM'] == [1.0, 2.0]
        assert decoded['Mb'] == [5.0, 6.0]
        assert decoded['allele_map'] == {"B": 0, "D": 1}
        assert decoded['founders'] == ["B", "D"]
        assert decoded['het_code'] == 2
        assert decoded['unk_code'] == 3
        assert decoded['dataset_name'] == "TestDS"
        assert decoded['cross_type'] == "riset"
        assert decoded['mat_allele'] == "B"
        assert decoded['pat_allele'] == "D"

    def test_decode_backwards_compat(self):
        """Should handle old stores that used 'positions' for single column."""
        old_data = {
            'markers': ['m1'],
            'positions': [1.5],
            'allele_map': {'B': 0},
            'founders': ['B', 'D'],
            'het_code': 2,
            'unk_code': 3,
        }
        decoded = decode_metadata(old_data)

        assert decoded['cM'] == [1.5]
        assert decoded['Mb'] == [1.5]
        assert decoded['markers'] == ['m1']

    def test_decode_empty_fallback(self):
        """Should return empty defaults for missing fields."""
        decoded = decode_metadata({})

        assert decoded['markers'] == []
        assert decoded['samples'] == []
        assert decoded['cM'] == []
        assert decoded['Mb'] == []
        assert decoded['allele_map'] == {}
        assert decoded['dataset_name'] == ''


class TestReconstructFromPayloads:
    """Test reconstruction in isolation — no LMDB needed."""

    @pytest.fixture
    def encoder(self):
        return DeltaEncoder()

    @pytest.fixture
    def sample_metadata(self):
        return {
            'markers': ['m1', 'm2'],
            'samples': ['s1', 's2'],
            'chromosomes': ['1', '1'],
            'cM': [1.0, 2.0],
            'Mb': [5.0, 6.0],
            'allele_map': {'B': 0, 'D': 1},
            'founders': ['B', 'D'],
            'het_code': 2,
            'unk_code': 3,
            'dataset_name': 'test',
            'cross_type': 'riset',
            'mat_allele': None,
            'pat_allele': None,
        }

    def test_full_snapshot_only(self, encoder, sample_metadata):
        """Reconstruct from a single full snapshot, no deltas."""
        matrix = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        payload = encoder.encode_full(matrix)

        plan = {
            'dataset_id': 'test',
            'target_version': 1,
            'full_version': 1,
            'needed_versions': [1],
        }
        payloads = {1: payload}

        result = reconstruct_from_payloads(plan, payloads, sample_metadata, encoder)

        assert np.array_equal(result.matrix, matrix)
        assert result.markers == ['m1', 'm2']
        assert result.samples == ['s1', 's2']
        assert result.cM == [1.0, 2.0]
        assert result.Mb == [5.0, 6.0]

    def test_with_sparse_delta(self, encoder, sample_metadata):
        """Reconstruct v2 from full snapshot + sparse delta."""
        v1_matrix = np.zeros((2, 2), dtype=np.uint8)
        v2_matrix = v1_matrix.copy()
        v2_matrix[0, 0] = 1  # Single cell change

        full_payload = encoder.encode_full(v1_matrix)
        delta_payload = encoder.encode_delta(v1_matrix, v2_matrix, 2)

        plan = {
            'dataset_id': 'test',
            'target_version': 2,
            'full_version': 1,
            'needed_versions': [1, 2],
        }
        payloads = {1: full_payload, 2: delta_payload}

        result = reconstruct_from_payloads(plan, payloads, sample_metadata, encoder)

        assert np.array_equal(result.matrix, v2_matrix)
        assert result.matrix[0, 0] == 1
        assert result.matrix[0, 1] == 0

    def test_with_xor_delta(self, encoder, sample_metadata):
        """Reconstruct v2 from full snapshot + XOR delta (dense change)."""
        v1_matrix = np.zeros((2, 2), dtype=np.uint8)
        v2_matrix = np.ones((2, 2), dtype=np.uint8)

        full_payload = encoder.encode_full(v1_matrix)
        # Force XOR by making change ratio exceed threshold
        delta_payload = encoder.encode_delta(v1_matrix, v2_matrix, 2)
        # Verify it's actually XOR
        assert delta_payload[0] == encoder.TYPE_XOR

        plan = {
            'dataset_id': 'test',
            'target_version': 2,
            'full_version': 1,
            'needed_versions': [1, 2],
        }
        payloads = {1: full_payload, 2: delta_payload}

        result = reconstruct_from_payloads(plan, payloads, sample_metadata, encoder)

        assert np.array_equal(result.matrix, v2_matrix)

    def test_metadata_attached(self, encoder, sample_metadata):
        """Metadata should be attached to reconstructed matrix."""
        matrix = np.zeros((2, 2), dtype=np.uint8)
        payload = encoder.encode_full(matrix)

        plan = {
            'dataset_id': 'test',
            'target_version': 1,
            'full_version': 1,
            'needed_versions': [1],
        }
        payloads = {1: payload}

        result = reconstruct_from_payloads(plan, payloads, sample_metadata, encoder)

        assert result.dataset_name == 'test'
        assert result.cross_type == 'riset'
        assert result.allele_map == {'B': 0, 'D': 1}
        assert result.founders == ['B', 'D']
