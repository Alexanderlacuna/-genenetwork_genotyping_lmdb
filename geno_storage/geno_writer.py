""".geno file writer — export GenotypeMatrix to .geno format."""

from pathlib import Path
from typing import Union

from .geno_parser import GenotypeMatrix


def export_genotype_file(
    genotype: GenotypeMatrix,
    output_path: Union[str, Path],
    include_comments: bool = False
) -> Path:
    """Export a GenotypeMatrix to a .geno file.

    Args:
        genotype: The GenotypeMatrix to export.
        output_path: Where to write the file.
        include_comments: Whether to include header comments.

    Returns:
        Path to the written file.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Build reverse map: numeric code → symbol
    reverse_map = {v: k for k, v in genotype.allele_map.items()}
    reverse_map[genotype.het_code] = 'H'
    reverse_map[genotype.unk_code] = 'U'

    lines = []

    # Comments
    if include_comments:
        lines.append(f"# Exported from LMDB genotype storage")
        lines.append(f"# Dataset: {genotype.dataset_name}")
        lines.append(f"# Cross type: {genotype.cross_type}")
        lines.append(f"# Markers: {len(genotype.markers)}, Samples: {len(genotype.samples)}")
        if genotype.mat_allele and genotype.pat_allele:
            lines.append(f"# Founders: {genotype.mat_allele} x {genotype.pat_allele}")

    # Metadata
    lines.append(f"@name:{genotype.dataset_name}")
    lines.append(f"@type:{genotype.cross_type}")
    if genotype.mat_allele:
        lines.append(f"@mat:{genotype.mat_allele}")
    if genotype.pat_allele:
        lines.append(f"@pat:{genotype.pat_allele}")
    lines.append(f"@het:H")
    lines.append(f"@unk:U")

    # Header row: Chr, Marker, Mb, cM, then sample names
    # Use position as Mb (we don't distinguish Mb vs cM in storage)
    header = ["Chr", "Locus", "Mb", "cM"] + genotype.samples
    lines.append("\t".join(header))

    # Data rows
    for i, marker in enumerate(genotype.markers):
        chrom = str(genotype.chromosomes[i]) if i < len(genotype.chromosomes) else ""
        pos = f"{genotype.positions[i]:.6f}" if i < len(genotype.positions) else "0.0"
        # cM column — we don't store cM separately, use same as Mb
        row_symbols = [reverse_map.get(int(code), 'U') for code in genotype.matrix[i]]
        row = [chrom, marker, pos, pos] + row_symbols
        lines.append("\t".join(row))

    with open(out, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
        f.write("\n")

    return out
