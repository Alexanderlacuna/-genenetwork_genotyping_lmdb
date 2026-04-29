""".geno file parser with multi-founder support."""

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union
from subprocess import check_output


@dataclass
class GenotypeMatrix:
    """Genotype matrix with metadata."""
    matrix: np.ndarray
    markers: List[str]
    samples: List[str]
    chromosomes: List[str]
    cM: List[Optional[float]]
    Mb: List[Optional[float]]
    allele_map: Dict[str, int]
    founders: List[str]
    het_code: int
    unk_code: int
    dataset_name: str = ""
    cross_type: str = ""
    mat_allele: Optional[str] = None
    pat_allele: Optional[str] = None
    genome_build: str = ""
    file_metadata: Dict = field(default_factory=dict)
    
    def get_allele_frequencies(self) -> Dict[int, int]:
        """Count allele codes."""
        unique, counts = np.unique(self.matrix, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))
    
    def decode_row(self, row_idx: int) -> List[str]:
        """Decode numeric codes to symbols."""
        reverse_map = {v: k for k, v in self.allele_map.items()}
        reverse_map[self.het_code] = 'H'
        reverse_map[self.unk_code] = 'U'
        return [reverse_map.get(code, '?') for code in self.matrix[row_idx]]


class GenoParser:
    """Parse .geno files."""
    
    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)
        self.metadata: Dict[str, str] = {}
        self.file_info: Dict[str, str] = {}
        self.header: List[str] = []
        self.raw_lines: List[str] = []
        
    def parse(self) -> GenotypeMatrix:
        """Parse file and return GenotypeMatrix."""
        self._parse_headers()
        encoding = self._detect_encoding()
        return self._read_data(encoding)
    
    def _parse_headers(self) -> None:
        """Parse @metadata and #comments."""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line.startswith('#'):
                    # Skip all # comments. Only parse key:value pairs if present.
                    if ':' in line:
                        content = line[1:].strip()
                        key, value = content.split(':', 1)
                        self.file_info[key.strip()] = value.strip()
                    continue
                elif line.startswith('@') and ':' in line:
                    content = line[1:]
                    key, value = content.split(':', 1)
                    self.metadata[key.strip()] = value.strip()
                    continue
                else:
                    self.header = line.split('\t')
                    break

            self.raw_lines = [l.strip() for l in f if l.strip()]
    
    def _detect_encoding(self) -> Dict:
        """Detect allele encoding scheme."""
        mat = self.metadata.get('mat')
        pat = self.metadata.get('pat')
        het = self.metadata.get('het', 'H')
        unk = self.metadata.get('unk', 'U')
        cross_type = self.metadata.get('type', '').lower()
        
        # Sample data for detection
        sample_alleles = set()
        for line in self.raw_lines[:100]:
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
    
    def _read_data(self, encoding: Dict) -> GenotypeMatrix:
        """Read data with encoding."""
        allele_map = encoding['allele_map'].copy()
        het_symbol = encoding['het_symbol']
        unk_symbol = encoding['unk_symbol']
        
        num_founders = len(encoding['founders'])
        het_code = num_founders
        unk_code = num_founders + 1
        
        allele_map[het_symbol] = het_code
        allele_map[unk_symbol] = unk_code
        
        if len(self.header) >= 4:
            samples = self.header[4:]
        else:
            samples = self.header
        
        n_markers = len(self.raw_lines)
        n_samples = len(samples)
        matrix = np.zeros((n_markers, n_samples), dtype=np.uint8)
        
        markers, chromosomes, cm_vals, mb_vals = [], [], [], []
        
        for i, line in enumerate(self.raw_lines):
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
            dataset_name=self.metadata.get('name', self.file_path.stem),
            cross_type=encoding['cross_type'],
            mat_allele=self.metadata.get('mat'),
            pat_allele=self.metadata.get('pat'),
            file_metadata=self.file_info
        )
    
    @staticmethod
    def count_lines(file_path: Union[str, Path]) -> int:
        """Count lines in file."""
        result = check_output(['wc', '-l', str(file_path)])
        return int(result.split()[0])


def parse_genotype_file(file_path: Union[str, Path]) -> GenotypeMatrix:
    """Parse a genotype file."""
    parser = GenoParser(file_path)
    return parser.parse()
