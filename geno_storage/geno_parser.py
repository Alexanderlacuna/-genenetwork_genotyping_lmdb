""".geno file parser with multi-founder support.

Refactored to separate actions (file I/O) from calculations
(encoding detection, data reading). Calculations are pure
functions with explicit inputs and outputs.

The only action is parse_genotype_file() which opens a file
and delegates to pure calculations.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union
from subprocess import check_output

import numpy as np

from .models import GenotypeMatrix


@dataclass
class ParsedHeader:
    """Result of parsing file headers — immutable data."""
    metadata: Dict[str, str]
    file_info: Dict[str, str]
    header: List[str]
    raw_lines: List[str]


def detect_encoding(metadata: Dict[str, str], raw_lines: List[str]) -> Dict:
    """Detect allele encoding scheme.

    Pure calculation: output depends only on explicit inputs.
    No file I/O, no mutable state.
    """
    mat = metadata.get('mat')
    pat = metadata.get('pat')
    het = metadata.get('het', 'H')
    unk = metadata.get('unk', 'U')
    cross_type = metadata.get('type', '').lower()

    # Sample data for detection
    sample_alleles = set()
    for line in raw_lines[:100]:
        parts = line.split('\t')
        if len(parts) > 4:
            sample_alleles.update(c for c in parts[4:] if c and c not in ['-', ''])

    allele_map = {}
    founders = []

    # Numeric genotypes
    numeric_alleles = sample_alleles & {'0', '1', '2', '3'}
    if numeric_alleles and len(numeric_alleles) > 1:
        allele_map = {'0': 0, '1': 1, '2': 2, '3': 3}
        founders = ['0', '1']
        het = '2'
        unk = '3'

    # 2-founder RI
    elif cross_type in ['riset', 'risib', 'consomic']:
        if mat and pat:
            allele_map = {mat: 0, pat: 1}
            founders = [mat, pat]
        else:
            allele_map = {'B': 0, 'D': 1}
            founders = ['B', 'D']

    # F2 intercross
    elif cross_type in ['f2', 'intercross']:
        if mat and pat:
            allele_map = {f'{mat}{mat}': 0, f'{mat}{pat}': 1, f'{pat}{mat}': 1, f'{pat}{pat}': 2}
            founders = [mat, pat]
        else:
            allele_map = {'AA': 0, 'AB': 1, 'BA': 1, 'BB': 2}
            founders = ['A', 'B']

    # Multi-founder
    elif cross_type in ['hs', 'do', 'diversity', 'outbred']:
        letter_alleles = sorted([a for a in sample_alleles if len(a) == 1 and a.isalpha()])
        if len(letter_alleles) >= 2:
            founders = letter_alleles
            for i, allele in enumerate(founders):
                allele_map[allele] = i
        else:
            founders = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
            for i, f in enumerate(founders):
                allele_map[f] = i

    # Auto-detect
    else:
        letter_alleles = sorted([a for a in sample_alleles if len(a) == 1 and a.isalpha()])
        if mat and pat:
            allele_map = {mat: 0, pat: 1}
            founders = [mat, pat]
        elif len(letter_alleles) >= 2:
            founders = letter_alleles
            for i, allele in enumerate(founders):
                allele_map[allele] = i
        else:
            allele_map = {'B': 0, 'D': 1}
            founders = ['B', 'D']

    return {
        'allele_map': allele_map,
        'founders': founders,
        'het_symbol': het,
        'unk_symbol': unk,
        'cross_type': cross_type or 'unknown'
    }


def read_data(
    parsed: ParsedHeader,
    encoding: Dict,
    dataset_name_default: str
) -> GenotypeMatrix:
    """Read genotype data from parsed lines.

    Pure calculation: assembles GenotypeMatrix from explicit inputs.
    No file I/O, no mutable state.
    """
    allele_map = encoding['allele_map'].copy()
    het_symbol = encoding['het_symbol']
    unk_symbol = encoding['unk_symbol']

    num_founders = len(encoding['founders'])
    het_code = num_founders
    unk_code = num_founders + 1

    allele_map[het_symbol] = het_code
    allele_map[unk_symbol] = unk_code

    header = parsed.header
    raw_lines = parsed.raw_lines
    metadata = parsed.metadata
    file_info = parsed.file_info

    if len(header) >= 4:
        samples = header[4:]
    else:
        samples = header

    n_markers = len(raw_lines)
    n_samples = len(samples)
    matrix = np.zeros((n_markers, n_samples), dtype=np.uint8)

    markers, chromosomes, cm_vals, mb_vals = [], [], [], []

    for i, line in enumerate(raw_lines):
        parts = line.split('\t')
        if len(parts) < 4:
            continue

        chromosomes.append(parts[0])
        markers.append(parts[1])

        # Parse cM (column 3 in file, index 2)
        if parts[2].strip():
            try:
                cm = float(parts[2])
            except ValueError:
                cm = None
        else:
            cm = None
        cm_vals.append(cm)

        # Parse Mb (column 4 in file, index 3)
        if len(parts) > 3 and parts[3].strip():
            try:
                mb = float(parts[3])
            except ValueError:
                mb = None
        else:
            mb = None
        mb_vals.append(mb)

        data = parts[4:] if len(parts) > 4 else []

        for j, allele in enumerate(data):
            if j >= n_samples:
                break

            if allele in allele_map:
                matrix[i, j] = allele_map[allele]
            elif allele.isdigit():
                matrix[i, j] = int(allele)
            elif allele == '-':
                matrix[i, j] = unk_code
            else:
                allele_upper = allele.upper()
                matrix[i, j] = allele_map.get(allele_upper, unk_code)

    clean_map = {k: v for k, v in allele_map.items() if k not in [het_symbol, unk_symbol]}

    return GenotypeMatrix(
        matrix=matrix,
        markers=markers,
        samples=samples,
        chromosomes=chromosomes,
        cM=cm_vals,
        Mb=mb_vals,
        allele_map=clean_map,
        founders=encoding['founders'],
        het_code=het_code,
        unk_code=unk_code,
        dataset_name=metadata.get('name', dataset_name_default),
        cross_type=encoding['cross_type'],
        mat_allele=metadata.get('mat'),
        pat_allele=metadata.get('pat'),
        file_metadata=file_info
    )


def parse_headers(file_handle) -> ParsedHeader:
    """Parse @metadata, #comments, and header row from file.

    Pure calculation relative to the file handle: given an open readable
    stream, return a ParsedHeader data structure. Does not mutate external
    state.
    """
    metadata: Dict[str, str] = {}
    file_info: Dict[str, str] = {}
    header: List[str] = []

    for line in file_handle:
        line = line.strip()
        if not line:
            continue

        if line.startswith('#'):
            if ':' in line:
                content = line[1:].strip()
                key, value = content.split(':', 1)
                file_info[key.strip()] = value.strip()
            continue
        elif line.startswith('@') and ':' in line:
            content = line[1:]
            key, value = content.split(':', 1)
            metadata[key.strip()] = value.strip()
            continue
        else:
            header = line.split('\t')
            break

    raw_lines = [l.strip() for l in file_handle if l.strip()]

    return ParsedHeader(
        metadata=metadata,
        file_info=file_info,
        header=header,
        raw_lines=raw_lines
    )


def count_lines(file_path: Union[str, Path]) -> int:
    """Count lines in file.

    Action: shell out to wc. Kept as a standalone function.
    """
    result = check_output(['wc', '-l', str(file_path)])
    return int(result.split()[0])


def parse_genotype_file(file_path: Union[str, Path]) -> GenotypeMatrix:
    """Parse a genotype file.

    Action: opens file, delegates to pure calculations, returns data.
    """
    path = Path(file_path)
    with open(path, 'r', encoding='utf-8') as f:
        parsed = parse_headers(f)

    encoding = detect_encoding(parsed.metadata, parsed.raw_lines)
    return read_data(parsed, encoding, path.stem)
