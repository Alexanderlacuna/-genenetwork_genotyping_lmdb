
# Usage Guide: Testing and CLI with Guix
## Description 

This is an Implementation of a  LMDB-based storage for genotype matrices to support fast, scalable access to large genetoype datasets.

The system should support efficient storage and retrieval of genotype data along with metadata (samples, markers, positions, chromosomes),
Include versioning of datasets and cryptographic v
erification to ensure data integrity across updates.



## TL;DR (Quick Reference)

```bash
# Enter environment
guix shell python-wrapper python-click python-lmdb python-numpy

# Import dataset
python lmdb_matrix.py import-genotype file.geno ./lmdb_store

# Force overwrite
python lmdb_matrix.py import-genotype file.geno ./lmdb_store --force

# Update dataset (new version)
python lmdb_matrix.py update-genotype DATASET file.geno ./lmdb_store

# List datasets
python lmdb_matrix.py list-datasets ./lmdb_store

# List versions

python lmdb_matrix.py list-versions DATASET ./lmdb_store

# Verify integrity

python lmdb_matrix.py verify DATASET ./lmdb_store

# Reconstruct latest
python lmdb_matrix.py reconstruct DATASET ./lmdb_store

# Reconstruct specific version
python lmdb_matrix.py reconstruct DATASET ./lmdb_store --version 2
````
````

> For detailed explanations, examples, and outputs, see sections 



## Prerequisites

You need Guix installed.

---

## Quick Start

### 1. Clone / Navigate to the Repository

### 2. Launch Guix Shell with Dependencies

```bash
guix shell python-wrapper python-click python-lmdb python-numpy python-pytest
```

### 3. Container (Isolated)

```bash
guix shell --container \
  --manifest=manifest.scm \
  --network \
  --share=$HOME/genotyping \
  -- \
  bash
```

---

## CLI Tool Usage

### Importing a Genotype File

#### Basic import

```bash
guix shell python-wrapper python-click python-lmdb python-numpy -- \
  python lmdb_matrix.py import-genotype \
  ~/genotype_files/genotype/BXD.geno \
  ./lmdb_store
```

#### With custom dataset ID and metadata

```bash
guix shell python-wrapper python-click python-lmdb python-numpy -- \
  python lmdb_matrix.py import-genotype \
  ~/genotype_files/genotype/BXD.geno \
  ./lmdb_store \
  --dataset-id "BXD_2018" \
  --author "Arthur Centeno" \
  --reason "Initial import from GeneNetwork"
```

#### Expected output

```text
Parsing BXD.geno...
  Dataset: BXD
  Type: riset
  Founders: ['B', 'D']
  Matrix: 7343 markers x 198 samples

✓ Stored as version 1
  Dataset ID: BXD_2018
  Hash: a3f7b2c8d9e1f456...
  Storage type: full
  Timestamp: 2024-01-15T10:30:00Z
```

---

### Importing an Existing Dataset

```bash
guix shell python-wrapper python-click python-lmdb python-numpy -- \
  python lmdb_matrix.py import-genotype \
  ~/genotype_files/genotype/BXD.geno \
  ./lmdb_store
```

#### Output when dataset exists

```text
Parsing BXD.geno...
  Dataset: BXD
  Type: riset
  Founders: ['B', 'D']
  Matrix: 7343 markers x 198 samples

⚠ Dataset 'BXD' already exists!
  Current version: 3
  Hash: b8e4c1d2a5f3e789...

Use --force to import anyway, or use 'update-genotype' to create a new version.
```

---

### Forcing an Import (`--force`)

If you want to overwrite an existing dataset **without creating a new version**, use the `--force` flag.

**Warning:** This will **replace the existing dataset** and may remove previous history.

```bash
guix shell python-wrapper python-click python-lmdb python-numpy -- \
  python lmdb_matrix.py import-genotype \
  ~/genotype_files/genotype/BXD.geno \
  ./lmdb_store \
  --force
```

#### When to use `--force`

* Re-importing after fixing parsing issues
* Resetting during development/testing
* Replacing corrupted initial imports

#### When **not** to use it

* When you need version history → use `update-genotype`
* When making incremental changes

#### Expected behavior

```text
⚠ Dataset 'BXD' already exists!
Forcing overwrite...

✓ Dataset replaced successfully
  Dataset ID: BXD
  Version reset to: v1
  Storage type: full
```

---

### `import-genotype` vs `update-genotype`

| Command           | Creates     | Version     | Storage              | History      |
| ----------------- | ----------- | ----------- | -------------------- | ------------ |
| `import-genotype` | New dataset | v1 (always) | Full matrix          | Starts fresh |
| `update-genotype` | New version | v2, v3, ... | Delta (changes only) | Preserved    |

---

### Updating a Dataset (Creating New Versions)

```bash
guix shell python-wrapper python-click python-lmdb python-numpy -- \
  python lmdb_matrix.py update-genotype \
  BXD_2018 \
  ~/genotype_files/genotype/BXD_corrected.geno \
  ./lmdb_store \
  --author "qc_pipeline" \
  --reason "QC correction"
```

---

### Hash Chain

```text
hash_v1 = SHA256("MATRIX_V1" + v1_data + null)
hash_v2 = SHA256("MATRIX_V1" + v2_delta + hash_v1)
hash_v3 = SHA256("MATRIX_V1" + v3_delta + hash_v2)
```

---

## Running Tests

### All Tests

```bash
guix shell python-wrapper python-lmdb python-numpy python-pytest -- \
  python -m pytest tests/ -v
```

### Specific Tests

```bash
guix shell python-wrapper python-numpy python-pytest -- \
  python -m pytest tests/test_hashing.py -v
```

