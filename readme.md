# LMDB Genotype Matrix Storage System

## Description

This project implements **LMDB-based storage** for genotype matrices, enabling fast and scalable access to large genomic datasets. It supports efficient storage and retrieval of genotype data alongside rich metadata — including samples, markers, positions, and chromosomes — while providing built-in dataset versioning and cryptographic integrity verification to ensure data consistency across updates.


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

# List datasets / versions
python lmdb_matrix.py list-datasets ./lmdb_store
python lmdb_matrix.py list-versions DATASET ./lmdb_store

# Verify integrity & Reconstruction
python lmdb_matrix.py verify DATASET ./lmdb_store
python lmdb_matrix.py reconstruct DATASET ./lmdb_store
```

---

## Prerequisites

- GNU Guix must be installed on your system.

---

## Environment Setup

### 1. Launch Guix Shell

To run the CLI tool with all necessary dependencies:

```bash
guix shell python-wrapper python-click python-lmdb python-numpy python-pytest
```

### 2. Isolated Container Mode

For a fully isolated environment (recommended for production/reproducibility):

```bash
guix shell --container \
  --manifest=manifest.scm \
  --network \
  --share=$HOME/genotyping \
  -- bash
```

---

## CLI Tool Usage

### Importing a Genotype File

#### Basic Import

```bash
guix shell python-wrapper python-click python-lmdb python-numpy -- \
  python lmdb_matrix.py import-genotype \
  ~/genotype_files/genotype/BXD.geno \
  ./lmdb_store
```

#### With Metadata

```bash
guix shell python-wrapper python-click python-lmdb python-numpy -- \
  python lmdb_matrix.py import-genotype \
  ~/genotype_files/genotype/BXD.geno \
  ./lmdb_store \
  --dataset-id "BXD_2018" \
  --author "Arthur Centeno" \
  --reason "Initial import from GeneNetwork"
```

### Handling Existing Datasets

If a dataset already exists, the tool will warn you. You have two options:

1. **Force Overwrite (`--force`):** Replaces the dataset and resets history to v1.
2. **Update (`update-genotype`):** Creates a new version (v2, v3, etc.) using delta storage.

| Command | Action | Versioning | Storage Type |
|---|---|---|---|
| `import-genotype` | New dataset | Always v1 | Full matrix |
| `update-genotype` | New version | Incremental | Delta (changes only) |

---

## Versioning & Data Integrity

The system uses a Hash Chain to verify data:

- `hash_v1 = SHA256("MATRIX_V1" + v1_data + null)`
- `hash_v2 = SHA256("MATRIX_V1" + v2_delta + hash_v1)`

---

## Running Tests

### Execute All Tests

```bash
guix shell python-wrapper python-lmdb python-numpy python-pytest -- \
  python -m pytest tests/ -v
```

### Run Specific Test Suite

```bash
guix shell python-wrapper python-numpy python-pytest -- \
  python -m pytest tests/test_hashing.py -v
```
