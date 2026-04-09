#!/usr/bin/env python3
"""CLI for versioned genotype storage."""

import sys
import json
from pathlib import Path
from datetime import datetime
from dataclasses import asdict

# Check for required dependencies
try:
    import click
    import lmdb
    import numpy as np
except ImportError as e:
    print(f"Error: Missing dependency {e}", file=sys.stderr)
    print("Run with: guix shell python-click python-lmdb python-numpy -- python lmdb_matrix.py ...", file=sys.stderr)
    sys.exit(1)

from geno_storage.geno_parser import parse_genotype_file
from geno_storage.matrix_store import MatrixStore


@click.group()
def cli():
    """Provably Verifiable Genotype Storage with LMDB"""
    pass


@cli.command(name='import-genotype')
@click.argument('geno_file', type=click.Path(exists=True))
@click.argument('lmdb_path', type=click.Path())
@click.option('--dataset-id', '-d', help='Dataset ID (default: from file metadata)')
@click.option('--author', '-a', default='cli_import', help='Author of import')
@click.option('--reason', '-r', default=None, help='Reason for import')
@click.option('--force', '-f', is_flag=True, help='Import even if dataset exists')
def import_genotype(geno_file, lmdb_path, dataset_id, author, reason, force):
    """Import a .geno file into LMDB with versioning."""
    geno_path = Path(geno_file)
    
    click.echo(f"Parsing {geno_path.name}...")
    genotype = parse_genotype_file(geno_path)
    
    click.echo(f"  Dataset: {genotype.dataset_name}")
    click.echo(f"  Type: {genotype.cross_type}")
    click.echo(f"  Founders: {genotype.founders}")
    click.echo(f"  Matrix: {genotype.matrix.shape[0]} markers x {genotype.matrix.shape[1]} samples")
    
    ds_id = dataset_id or genotype.dataset_name
    reason = reason or f"Import from {geno_path.name}"
    
    with MatrixStore(lmdb_path) as store:
        # Check if dataset already exists
        try:
            existing_ver, existing_hash = store.get_current_version(ds_id)
            if not force:
                click.echo(f"\n⚠ Dataset '{ds_id}' already exists!")
                click.echo(f"  Current version: {existing_ver}")
                click.echo(f"  Hash: {existing_hash}")
                click.echo(f"\nUse --force to import anyway, or use 'update-genotype' to create a new version.")
                return
            else:
                click.echo(f"\n⚠ Overwriting existing dataset '{ds_id}' (version {existing_ver})")
        except ValueError:
            pass  # Dataset doesn't exist, proceed with import
        
        version = store.store_initial(
            ds_id,
            genotype,
            author=author,
            reason=reason
        )
        
        click.echo(f"\n✓ Stored as version {version.matrix_version}")
        click.echo(f"  Dataset ID: {ds_id}")
        click.echo(f"  Hash: {version.matrix_hash}")
        click.echo(f"  Storage type: {version.storage_type}")
        click.echo(f"  Timestamp: {version.timestamp}")


@cli.command(name='update-genotype')
@click.argument('dataset_id')
@click.argument('geno_file', type=click.Path(exists=True))
@click.argument('lmdb_path', type=click.Path())
@click.option('--author', '-a', required=True, help='Author of update')
@click.option('--reason', '-r', required=True, help='Reason for update')
def update_genotype(dataset_id, geno_file, lmdb_path, author, reason):
    """Update an existing dataset with a new genotype file."""
    geno_path = Path(geno_file)
    
    click.echo(f"Parsing {geno_path.name}...")
    genotype = parse_genotype_file(geno_path)
    
    click.echo(f"  Matrix: {genotype.matrix.shape}")
    
    with MatrixStore(lmdb_path) as store:
        version = store.store_update(
            dataset_id,
            genotype.matrix,
            author=author,
            reason=reason
        )
        
        click.echo(f"\n✓ Created version {version.matrix_version}")
        click.echo(f"  Hash: {version.matrix_hash}")
        click.echo(f"  Storage type: {version.storage_type}")
        click.echo(f"  Previous: {version.prev_matrix_hash}")


@cli.command(name='list-datasets')
@click.argument('lmdb_path', type=click.Path(exists=True))
def list_datasets(lmdb_path):
    """List all datasets in the LMDB store."""
    with MatrixStore(lmdb_path, read_only=True) as store:
        with store.env.begin() as txn:
            geno_db = store._get_db(txn, store.DB_GENOTYPES)
            cursor = txn.cursor(db=geno_db)
            
            datasets = []
            for key, value in cursor:
                dataset_id = key.decode('utf-8')
                data = json.loads(value.decode('utf-8'))
                datasets.append({
                    'id': dataset_id,
                    'version': data.get('current_version', data.get('current_matrix_version', '?')),
                    'hash': data.get('current_hash', data.get('current_matrix_hash', '?'))[:16],
                    'shape': f"{data.get('nrows', '?')}x{data.get('ncols', '?')}"
                })
            
            if not datasets:
                click.echo("No datasets found in LMDB store.")
                return
            
            click.echo(f"\nDatasets in {lmdb_path}:")
            click.echo("-" * 70)
            click.echo(f"{'Dataset ID':<20} {'Version':<10} {'Hash':<20} {'Shape'}")
            click.echo("-" * 70)
            
            for ds in sorted(datasets, key=lambda x: x['id']):
                click.echo(
                    f"{ds['id']:<20} "
                    f"v{ds['version']:<9} "
                    f"{ds['hash']:<20} "
                    f"{ds['shape']}"
                )
            
            click.echo("-" * 70)
            click.echo(f"Total: {len(datasets)} dataset(s)")


@cli.command(name='list-versions')
@click.argument('dataset_id')
@click.argument('lmdb_path', type=click.Path(exists=True))
def list_versions(dataset_id, lmdb_path):
    """List all versions for a dataset."""
    with MatrixStore(lmdb_path, read_only=True) as store:
        versions = store.list_versions(dataset_id)
        
        if not versions:
            click.echo(f"No versions found for dataset: {dataset_id}")
            return
        
        click.echo(f"\nDataset: {dataset_id}")
        click.echo("-" * 100)
        click.echo(f"{'Ver':<5} {'Type':<8} {'Hash':<18} {'Timestamp':<25} {'Author':<15} {'Reason'}")
        click.echo("-" * 100)
        
        for v in versions:
            click.echo(
                f"{v['matrix_version']:<5} "
                f"{v['storage_type']:<8} "
                f"{v['matrix_hash'][:16]:<18} "
                f"{v['timestamp']:<25} "
                f"{v['author']:<15} "
                f"{v['reason'][:40]}"
            )
        
        click.echo("-" * 100)
        click.echo(f"Total: {len(versions)} versions")


@cli.command(name='verify')
@click.argument('dataset_id')
@click.argument('lmdb_path', type=click.Path(exists=True))
def verify_dataset(dataset_id, lmdb_path):
    """Verify the cryptographic integrity of a dataset."""
    click.echo(f"Verifying dataset: {dataset_id}")
    click.echo("Checking hash chain...")
    
    with MatrixStore(lmdb_path, read_only=True) as store:
        valid, errors = store.verify_dataset(dataset_id)
        
        if valid:
            click.echo("✓ Verification PASSED")
            click.echo("  Hash chain is intact")
            click.echo("  All versions verified")
            
            # Show chain info
            versions = store.list_versions(dataset_id)
            click.echo(f"\n  Versions: {len(versions)}")
            click.echo(f"  Head: v{versions[-1]['matrix_version']} ({versions[-1]['matrix_hash'][:16]}...)")
        else:
            click.echo("✗ Verification FAILED")
            for error in errors:
                click.echo(f"  - {error}")
            sys.exit(1)


@cli.command(name='reconstruct')
@click.argument('dataset_id')
@click.argument('lmdb_path', type=click.Path(exists=True))
@click.option('--version', '-v', type=int, default=None, help='Version to reconstruct (default: current)')
@click.option('--output', '-o', type=click.Path(), help='Output numpy file (.npy)')
@click.option('--info-only', is_flag=True, help='Only show matrix info, don\'t print full matrix')
def reconstruct(dataset_id, lmdb_path, version, output, info_only):
    """Reconstruct a specific version of a matrix."""
    with MatrixStore(lmdb_path, read_only=True) as store:
        target_ver = version
        if target_ver is None:
            target_ver, _ = store.get_current_version(dataset_id)
        
        click.echo(f"Reconstructing {dataset_id} version {target_ver}...")
        
        genotype = store.get_matrix(dataset_id, target_ver)
        matrix = genotype.matrix
        
        click.echo(f"\nMatrix info:")
        click.echo(f"  Shape: {matrix.shape}")
        click.echo(f"  Dtype: {matrix.dtype}")
        click.echo(f"  Size: {matrix.nbytes / 1024:.2f} KB")
        click.echo(f"  Samples: {len(genotype.samples)}")
        click.echo(f"  Markers: {len(genotype.markers)}")
        
        if not info_only:
            click.echo(f"\nFirst 10x10:")
            rows = min(10, matrix.shape[0])
            cols = min(10, matrix.shape[1])
            click.echo(matrix[:rows, :cols])
        
        if output:
            np.save(output, matrix)
            click.echo(f"\n✓ Saved to {output}")


@cli.command(name='import-directory')
@click.argument('genotype_directory', type=click.Path(exists=True))
@click.argument('lmdb_path', type=click.Path())
@click.option('--author', '-a', default='batch_import', help='Author')
def import_directory(genotype_directory, lmdb_path, author):
    """Import all .geno files from a directory."""
    directory = Path(genotype_directory)
    geno_files = sorted(directory.glob('*.geno'))
    
    if not geno_files:
        click.echo(f"No .geno files found in {directory}")
        return
    
    click.echo(f"Found {len(geno_files)} files to import")
    click.echo()
    
    success = 0
    failed = 0
    
    for geno_file in geno_files:
        try:
            click.echo(f"Importing {geno_file.name}...")
            genotype = parse_genotype_file(geno_file)
            
            with MatrixStore(lmdb_path) as store:
                version = store.store_initial(
                    genotype.dataset_name,
                    genotype,
                    author=author,
                    reason=f"Batch import from {geno_file.name}"
                )
                
                click.echo(f"  ✓ Version {version.matrix_version} ({genotype.matrix.shape})")
                success += 1
                
        except Exception as e:
            click.echo(f"  ✗ Failed: {e}")
            failed += 1
    
    click.echo()
    click.echo(f"Import complete: {success} succeeded, {failed} failed")


if __name__ == '__main__':
    cli()
