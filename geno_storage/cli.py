"""
Command-line interface for genotype storage.

Similar to lmdb_matrix.py but with versioning support.
"""

import argparse
import sys
from pathlib import Path

from .geno_parser import parse_genotype_file
from .matrix_store import MatrixStore


def cmd_import(args):
    """Import a genotype file into LMDB with versioning."""
    geno_file = Path(args.geno_file)
    if not geno_file.exists():
        print(f"Error: File not found: {geno_file}", file=sys.stderr)
        return 1
    
    # Parse genotype file
    print(f"Parsing {geno_file.name}...")
    genotype = parse_genotype_file(geno_file)
    
    print(f"  Dataset: {genotype.dataset_name}")
    print(f"  Cross type: {genotype.cross_type}")
    print(f"  Founders: {genotype.founders}")
    print(f"  Matrix size: {genotype.matrix.shape}")
    print(f"  Markers: {len(genotype.markers)}")
    print(f"  Samples: {len(genotype.samples)}")
    
    # Store in LMDB
    with MatrixStore(args.lmdb_path) as store:
        dataset_id = args.dataset_id or genotype.dataset_name
        
        version = store.store_initial(
            dataset_id,
            genotype,
            author=args.author or "cli_import",
            reason=args.reason or f"Import from {geno_file.name}"
        )
        
        print(f"\n✓ Stored as version {version.matrix_version}")
        print(f"  Hash: {version.matrix_hash[:16]}...")
        print(f"  Storage: {version.storage_type}")
        print(f"  LMDB path: {args.lmdb_path}")
    
    return 0


def cmd_update(args):
    """Update an existing dataset with a new genotype file."""
    geno_file = Path(args.geno_file)
    if not geno_file.exists():
        print(f"Error: File not found: {geno_file}", file=sys.stderr)
        return 1
    
    genotype = parse_genotype_file(geno_file)
    
    with MatrixStore(args.lmdb_path) as store:
        version = store.store_update(
            args.dataset_id,
            genotype.matrix,
            author=args.author or "cli_update",
            reason=args.reason or f"Update from {geno_file.name}"
        )
        
        print(f"✓ Created version {version.matrix_version}")
        print(f"  Hash: {version.matrix_hash[:16]}...")
        print(f"  Storage: {version.storage_type}")
    
    return 0


def cmd_list(args):
    """List versions for a dataset."""
    with MatrixStore(args.lmdb_path, read_only=True) as store:
        versions = store.list_versions(args.dataset_id)
        
        if not versions:
            print(f"No versions found for dataset: {args.dataset_id}")
            return 1
        
        print(f"\nDataset: {args.dataset_id}")
        print("-" * 80)
        print(f"{'Version':<8} {'Type':<8} {'Hash (trunc)':<20} {'Author':<15} {'Reason'}")
        print("-" * 80)
        
        for v in versions:
            print(f"{v['matrix_version']:<8} {v['storage_type']:<8} "
                  f"{v['matrix_hash'][:18]:<20} {v['author']:<15} {v['reason'][:30]}")
        
        print("-" * 80)
        print(f"Total versions: {len(versions)}")
    
    return 0


def cmd_verify(args):
    """Verify hash chain integrity."""
    with MatrixStore(args.lmdb_path, read_only=True) as store:
        valid, errors = store.verify_dataset(args.dataset_id)
        
        if valid:
            print(f"✓ Dataset '{args.dataset_id}' verification PASSED")
            print("  Hash chain is intact")
        else:
            print(f"✗ Dataset '{args.dataset_id}' verification FAILED")
            for error in errors:
                print(f"  - {error}")
            return 1
    
    return 0


def cmd_reconstruct(args):
    """Reconstruct a specific version."""
    with MatrixStore(args.lmdb_path, read_only=True) as store:
        version = args.version
        if version is None:
            version, _ = store.get_current_version(args.dataset_id)
        
        print(f"Reconstructing version {version}...")
        matrix = store.get_matrix(args.dataset_id, version)
        
        print(f"\nMatrix shape: {matrix.shape}")
        print(f"Data type: {matrix.dtype}")
        print(f"\nFirst 5x5:")
        print(matrix[:5, :5])
        
        if args.output:
            import numpy as np
            np.save(args.output, matrix)
            print(f"\n✓ Saved to {args.output}")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Provably Verifiable Genotype Storage"
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Import command
    import_cmd = subparsers.add_parser(
        'import',
        help='Import a genotype file'
    )
    import_cmd.add_argument('geno_file', help='Path to .geno file')
    import_cmd.add_argument('lmdb_path', help='Path to LMDB database')
    import_cmd.add_argument('--dataset-id', help='Dataset ID (default: from file)')
    import_cmd.add_argument('--author', help='Author of import')
    import_cmd.add_argument('--reason', help='Reason for import')
    
    # Update command
    update_cmd = subparsers.add_parser(
        'update',
        help='Update existing dataset'
    )
    update_cmd.add_argument('dataset_id', help='Dataset ID')
    update_cmd.add_argument('geno_file', help='Path to updated .geno file')
    update_cmd.add_argument('lmdb_path', help='Path to LMDB database')
    update_cmd.add_argument('--author', required=True, help='Author of update')
    update_cmd.add_argument('--reason', required=True, help='Reason for update')
    
    # List command
    list_cmd = subparsers.add_parser(
        'list',
        help='List dataset versions'
    )
    list_cmd.add_argument('dataset_id', help='Dataset ID')
    list_cmd.add_argument('lmdb_path', help='Path to LMDB database')
    
    # Verify command
    verify_cmd = subparsers.add_parser(
        'verify',
        help='Verify dataset integrity'
    )
    verify_cmd.add_argument('dataset_id', help='Dataset ID')
    verify_cmd.add_argument('lmdb_path', help='Path to LMDB database')
    
    # Reconstruct command
    recon_cmd = subparsers.add_parser(
        'reconstruct',
        help='Reconstruct a specific version'
    )
    recon_cmd.add_argument('dataset_id', help='Dataset ID')
    recon_cmd.add_argument('lmdb_path', help='Path to LMDB database')
    recon_cmd.add_argument('--version', type=int, help='Version to reconstruct (default: current)')
    recon_cmd.add_argument('--output', help='Output .npy file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    commands = {
        'import': cmd_import,
        'update': cmd_update,
        'list': cmd_list,
        'verify': cmd_verify,
        'reconstruct': cmd_reconstruct,
    }
    
    return commands[args.command](args)


if __name__ == '__main__':
    sys.exit(main())
