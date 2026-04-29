"""Matrix reconstruction — pure calculations.

These functions assemble GenotypeMatrix from payloads and metadata.
No LMDB I/O, no side effects, no mutable state.
"""

from typing import Dict

from .models import GenotypeMatrix


def reconstruct_from_payloads(
    plan: Dict,
    payloads: Dict[int, bytes],
    metadata: Dict,
    delta_encoder
) -> GenotypeMatrix:
    """Reconstruct GenotypeMatrix from payloads.

    Pure calculation: given payloads, plan, and metadata, assemble matrix.
    No LMDB I/O, no side effects.

    Args:
        plan: Reconstruction plan dict with 'full_version', 'target_version'.
        payloads: Mapping version -> payload bytes.
        metadata: Metadata dict from full snapshot.
        delta_encoder: DeltaEncoder instance for decode/apply.

    Returns:
        Reconstructed GenotypeMatrix.
    """
    full_version = plan['full_version']
    target_version = plan['target_version']

    # Decode full snapshot
    _, current_matrix = delta_encoder.decode(payloads[full_version])

    # Apply deltas
    for v in range(full_version + 1, target_version + 1):
        _, delta_data = delta_encoder.decode(payloads[v])
        delta_type = payloads[v][0]
        current_matrix = delta_encoder.apply_delta(current_matrix, delta_data, delta_type)

    # Return GenotypeMatrix with metadata
    return GenotypeMatrix(
        matrix=current_matrix,
        markers=metadata['markers'],
        samples=metadata['samples'],
        chromosomes=metadata['chromosomes'],
        cM=metadata['cM'],
        Mb=metadata['Mb'],
        allele_map=metadata['allele_map'],
        founders=metadata['founders'],
        het_code=metadata['het_code'],
        unk_code=metadata['unk_code'],
        dataset_name=metadata['dataset_name'],
        cross_type=metadata['cross_type'],
        mat_allele=metadata['mat_allele'],
        pat_allele=metadata['pat_allele'],
    )
