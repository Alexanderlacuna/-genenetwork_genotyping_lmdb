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
from geno_storage.geno_writer import export_genotype_file
from geno_storage.matrix_store import MatrixStore
from geno_storage.cli_helpers import resolved_matrix


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
@click.option('--compression', '-c', default=None, help='Payload compression algorithm (default: none). Set once at store creation time. Choices: zlib, none.')
def import_genotype(geno_file, lmdb_path, dataset_id, author, reason, compression):
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
            click.echo(f"\n⚠ Dataset '{ds_id}' already exists!")
            click.echo(f"  Current version: {existing_ver}")
            click.echo(f"  Hash: {existing_hash}")
            click.echo(f"\nUse 'update-genotype' to create a new version, or remove the dataset from the LMDB store and re-import.")
            return
        except ValueError:
            pass  # Dataset doesn't exist, proceed with import

        version = store.store_initial(
            ds_id,
            genotype,
            author=author,
            reason=reason,
            compression=compression
        )

        click.echo(f"\n✓ Stored as version {version.matrix_version}")
        click.echo(f"  Dataset ID: {ds_id}")
        click.echo(f"  Hash: {version.matrix_hash}")
        click.echo(f"  Storage type: {version.storage_type}")
        if version.compression:
            click.echo(f"  Compression: {version.compression}")
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
        try:
            version = store.store_update(
                dataset_id,
                genotype,
                author=author,
                reason=reason
            )
        except ValueError as e:
            click.echo(f"\n✗ Update failed:\n  {e}")
            sys.exit(1)

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
            geno_db = store.get_db(txn, store.DB_GENOTYPES)
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
            click.echo(f"✗ Dataset '{dataset_id}' has no versions in this store.")
            click.echo("  Use 'list-datasets' to see available datasets.")
            sys.exit(1)
        
        click.echo(f"\nDataset: {dataset_id}")
        click.echo("-" * 110)
        click.echo(f"{'Ver':<5} {'Type':<8} {'Compress':<10} {'Hash':<18} {'Timestamp':<25} {'Author':<15} {'Reason'}")
        click.echo("-" * 110)
        
        for v in versions:
            compress_str = v.get('compression', '-') or '-'
            click.echo(
                f"{v['matrix_version']:<5} "
                f"{v['storage_type']:<8} "
                f"{compress_str:<10} "
                f"{v['matrix_hash'][:16]:<18} "
                f"{v['timestamp']:<25} "
                f"{v['author']:<15} "
                f"{v['reason'][:40]}"
            )
        
        click.echo("-" * 110)
        click.echo(f"Total: {len(versions)} versions")


@cli.command(name='verify')
@click.argument('dataset_id')
@click.argument('lmdb_path', type=click.Path(exists=True))
@click.option('--fast', is_flag=True, help='Fast mode: check chain linkage only (skip payload hashing)')
def verify_dataset(dataset_id, lmdb_path, fast):
    """Verify the cryptographic integrity of a dataset."""
    click.echo(f"Verifying dataset: {dataset_id}")

    with MatrixStore(lmdb_path, read_only=True) as store:
        versions = store.list_versions(dataset_id)

        if not versions:
            click.echo(f"\n✗ Dataset '{dataset_id}' has no versions in this store.")
            # Try to suggest available datasets
            try:
                all_ds = []
                with store.env.begin() as txn:
                    geno_db = store.get_db(txn, store.DB_GENOTYPES)
                    cursor = txn.cursor(db=geno_db)
                    for key, _ in cursor:
                        all_ds.append(key.decode('utf-8'))
                if all_ds:
                    click.echo(f"  Available datasets: {', '.join(sorted(all_ds))}")
            except Exception:
                pass
            click.echo("  Use 'list-datasets' to see all datasets.")
            sys.exit(1)

        if fast:
            click.echo("Mode: FAST (chain linkage only)")
            click.echo("Checking hash chain linkage...")
            valid, errors = store.verify_dataset_fast(dataset_id)

            if valid:
                click.echo("✓ Fast verification PASSED")
                click.echo("  Hash chain linkage is intact")
                click.echo("  NOTE: Payload integrity not checked (use without --fast for full verify)")
            else:
                click.echo("✗ Fast verification FAILED")
                for error in errors:
                    click.echo(f"  - {error}")
                sys.exit(1)
        else:
            click.echo("Mode: FULL (recomputing all hashes from payloads)")
            click.echo("Checking hash chain...")
            valid, errors = store.verify_dataset(dataset_id)

            if valid:
                click.echo("✓ Full verification PASSED")
                click.echo("  Hash chain is intact")
                click.echo("  All payloads verified")
            else:
                click.echo("✗ Full verification FAILED")
                for error in errors:
                    click.echo(f"  - {error}")
                sys.exit(1)

        click.echo(f"\n  Versions: {len(versions)}")
        click.echo(f"  Head: v{versions[-1]['matrix_version']} ({versions[-1]['matrix_hash'][:16]}...)")


@cli.command(name='reconstruct')
@click.argument('dataset_id')
@click.argument('lmdb_path', type=click.Path(exists=True))
@click.option('--version', '-v', type=int, default=None, help='Version to reconstruct (default: current)')
@click.option('--output', '-o', type=click.Path(), help='Output numpy file (.npy)')
@click.option('--info-only', is_flag=True, help='Only show matrix info, don\'t print full matrix')
def reconstruct(dataset_id, lmdb_path, version, output, info_only):
    """Reconstruct a specific version of a matrix."""
    try:
        with resolved_matrix(lmdb_path, dataset_id, version=version) as (_, target_ver, genotype):
            click.echo(f"Reconstructing {dataset_id} version {target_ver}...")
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
    except ValueError as e:
        click.echo(f"\n✗ {e}")
        sys.exit(1)


@cli.command(name='export-genotype')
@click.argument('dataset_id')
@click.argument('lmdb_path', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path())
@click.option('--version', '-v', type=int, default=None, help='Version to export (default: current)')
@click.option('--comments', is_flag=True, help='Include header comments in export')
def export_genotype(dataset_id, lmdb_path, output_file, version, comments):
    """Export a dataset version to a .geno file."""
    try:
        with resolved_matrix(lmdb_path, dataset_id, version=version) as (_, target_ver, genotype):
            click.echo(f"Exporting {dataset_id} version {target_ver}...")

            out_path = export_genotype_file(
                genotype,
                output_file,
                include_comments=comments
            )

            click.echo(f"\n✓ Exported to {out_path}")
            click.echo(f"  Markers: {len(genotype.markers)}")
            click.echo(f"  Samples: {len(genotype.samples)}")
            click.echo(f"  Shape: {genotype.matrix.shape}")
    except ValueError as e:
        click.echo(f"\n✗ Export failed:\n  {e}")
        sys.exit(1)


@cli.command(name='diff')
@click.argument('dataset_id')
@click.argument('lmdb_path', type=click.Path(exists=True))
@click.option('--from', 'from_version', type=int, required=True, help='Source version')
@click.option('--to', 'to_version', type=int, required=True, help='Target version')
@click.option('--max', 'max_rows', type=int, default=50, help='Max differences to show (default: 50)')
def diff_dataset(dataset_id, lmdb_path, from_version, to_version, max_rows):
    """Show differences between two versions."""
    with MatrixStore(lmdb_path, read_only=True) as store:
        click.echo(f"Loading version {from_version}...")
        try:
            geno_from = store.get_matrix(dataset_id, from_version)
        except ValueError as e:
            click.echo(f"\n✗ Failed to load version {from_version}:\n  {e}")
            sys.exit(1)
        click.echo(f"Loading version {to_version}...")
        try:
            geno_to = store.get_matrix(dataset_id, to_version)
        except ValueError as e:
            click.echo(f"\n✗ Failed to load version {to_version}:\n  {e}")
            sys.exit(1)

        if geno_from.matrix.shape != geno_to.matrix.shape:
            click.echo(f"✗ Shape mismatch: v{from_version} {geno_from.matrix.shape} vs v{to_version} {geno_to.matrix.shape}")
            sys.exit(1)

        diff_mask = geno_from.matrix != geno_to.matrix
        total_diffs = np.sum(diff_mask)

        if total_diffs == 0:
            click.echo(f"\n✓ No differences between v{from_version} and v{to_version}")
            return

        change_ratio = total_diffs / geno_from.matrix.size * 100
        click.echo(f"\nDifferences between v{from_version} and v{to_version}:")
        click.echo(f"  Total changed cells: {total_diffs} / {geno_from.matrix.size} ({change_ratio:.4f}%)")
        click.echo(f"  Changed markers (rows): {len(np.unique(np.where(diff_mask)[0]))}")
        click.echo(f"  Changed samples (cols): {len(np.unique(np.where(diff_mask)[1]))}")

        # Show first N differences with marker/sample names
        diffs = np.argwhere(diff_mask)
        show_count = min(max_rows, len(diffs))
        click.echo(f"\nFirst {show_count} changes:")
        click.echo(f"{'Marker':<20} {'Sample':<10} {'v'+str(from_version):<6} {'v'+str(to_version):<6}")
        click.echo("-" * 50)

        for idx, (row, col) in enumerate(diffs[:show_count]):
            marker = geno_from.markers[row] if row < len(geno_from.markers) else f"row_{row}"
            sample = geno_from.samples[col] if col < len(geno_from.samples) else f"col_{col}"
            val_from = geno_from.matrix[row, col]
            val_to = geno_to.matrix[row, col]
            click.echo(f"{marker:<20} {sample:<10} {val_from:<6} {val_to:<6}")

        if total_diffs > max_rows:
            click.echo(f"\n... and {total_diffs - max_rows} more changes")


@cli.command(name='stats')
@click.argument('dataset_id')
@click.argument('lmdb_path', type=click.Path(exists=True))
def stats_dataset(dataset_id, lmdb_path):
    """Show storage statistics for a dataset."""
    with MatrixStore(lmdb_path, read_only=True) as store:
        versions = store.list_versions(dataset_id)

        if not versions:
            click.echo(f"✗ Dataset '{dataset_id}' has no versions in this store.")
            click.echo("  Use 'list-datasets' to see available datasets.")
            sys.exit(1)

        full_count = sum(1 for v in versions if v['storage_type'] == 'full')
        delta_count = len(versions) - full_count

        # Get payload sizes
        total_payload = 0
        full_payload = 0
        delta_payload = 0
        with store.env.begin() as txn:
            matrix_db = store.get_db(txn, store.DB_MATRIX_HISTORY)
            for v in versions:
                key = store.make_history_key(dataset_id, v['matrix_version'])
                payload = txn.get(key + b':payload', db=matrix_db)
                if payload:
                    size = len(payload)
                    total_payload += size
                    if v['storage_type'] == 'full':
                        full_payload += size
                    else:
                        delta_payload += size

        # Estimate what full-only storage would be
        if full_count > 0:
            avg_full = full_payload / full_count
            hypothetical_all_full = avg_full * len(versions)
            savings = (1 - total_payload / hypothetical_all_full) * 100 if hypothetical_all_full > 0 else 0
        else:
            hypothetical_all_full = 0
            savings = 0

        click.echo(f"\nDataset: {dataset_id}")
        click.echo("-" * 50)
        click.echo(f"Total versions:       {len(versions)}")
        click.echo(f"Full snapshots:       {full_count}")
        click.echo(f"Deltas:               {delta_count}")
        click.echo(f"")
        click.echo(f"Total payload:        {total_payload / 1024:.2f} KB")
        click.echo(f"  Full snapshots:     {full_payload / 1024:.2f} KB")
        click.echo(f"  Deltas:             {delta_payload / 1024:.2f} KB")
        click.echo(f"")
        if hypothetical_all_full > 0:
            click.echo(f"Without deltas:       {hypothetical_all_full / 1024:.2f} KB")
            click.echo(f"Storage savings:      {savings:.1f}%")
        click.echo("-" * 50)

        # Show version breakdown
        click.echo(f"\nVersion breakdown:")
        click.echo(f"{'Ver':<5} {'Type':<8} {'Size (KB)':<12} {'Author':<15}")
        click.echo("-" * 45)
        with store.env.begin() as txn:
            matrix_db = store.get_db(txn, store.DB_MATRIX_HISTORY)
            for v in versions:
                key = store.make_history_key(dataset_id, v['matrix_version'])
                payload = txn.get(key + b':payload', db=matrix_db)
                size_kb = len(payload) / 1024 if payload else 0
                click.echo(f"{v['matrix_version']:<5} {v['storage_type']:<8} {size_kb:<12.2f} {v['author']:<15}")


@cli.command(name='import-directory')
@click.argument('genotype_directory', type=click.Path(exists=True))
@click.argument('lmdb_path', type=click.Path())
@click.option('--author', '-a', default='batch_import', help='Author')
@click.option('--compression', '-c', default=None, help='Payload compression algorithm (default: none). Set once at store creation time. Choices: zlib, none.')
def import_directory(genotype_directory, lmdb_path, author, compression):
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
                    reason=f"Batch import from {geno_file.name}",
                    compression=compression
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
