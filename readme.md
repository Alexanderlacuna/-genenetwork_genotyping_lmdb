# LMDB Genotype Matrix Storage System

Versioned, cryptographically verifiable genotype storage using LMDB.

---

## Quick Start

```bash
guix shell python-wrapper python-click python-lmdb python-numpy

python lmdb_matrix.py import-genotype file.geno ./lmdb_store
python lmdb_matrix.py update-genotype DATASET file.geno ./lmdb_store
python lmdb_matrix.py list-datasets ./lmdb_store
python lmdb_matrix.py list-versions DATASET ./lmdb_store
python lmdb_matrix.py verify DATASET ./lmdb_store          # full verify
python lmdb_matrix.py verify DATASET ./lmdb_store --fast   # fast verify
python lmdb_matrix.py reconstruct DATASET ./lmdb_store
python lmdb_matrix.py export-genotype DATASET ./lmdb_store output.geno
python lmdb_matrix.py diff DATASET ./lmdb_store --from 1 --to 2
python lmdb_matrix.py stats DATASET ./lmdb_store
```

**Compression is opt-in.** Use `--compression zlib` on `import-genotype` to enable zlib compression for a store. This setting is fixed at store creation time and applies to all datasets in that store. `update-genotype` reads the compression setting automatically — you never specify it again.

---

## Commands

### Import

```bash
python lmdb_matrix.py import-genotype file.geno ./lmdb_store \
  --dataset-id "BXD" \
  --author "Arthur Centeno" \
  --reason "Initial import"

# With compression (opt-in, set once at store creation)
python lmdb_matrix.py import-genotype file.geno ./lmdb_store \
  --compression zlib \
  --dataset-id "BXD"
```

### Update (creates new version)

```bash
python lmdb_matrix.py update-genotype BXD file.geno ./lmdb_store \
  --author "qc_pipeline" \
  --reason "QC correction"
```

### Verify

```bash
# Full verify: recomputes all hashes from payloads (thorough but slow)
python lmdb_matrix.py verify BXD ./lmdb_store

# Fast verify: checks chain linkage only (quick routine check)
python lmdb_matrix.py verify BXD ./lmdb_store --fast
```

### Reconstruct

```bash
# Current version
python lmdb_matrix.py reconstruct BXD ./lmdb_store

# Specific version
python lmdb_matrix.py reconstruct BXD ./lmdb_store --version 2 --output BXD_v2.npy
```

### Export to .geno

```bash
# Default: no header comments, clean output
python lmdb_matrix.py export-genotype BXD ./lmdb_store BXD_exported.geno

# With header comments
python lmdb_matrix.py export-genotype BXD ./lmdb_store BXD_exported.geno --comments

# Specific version
python lmdb_matrix.py export-genotype BXD ./lmdb_store BXD_v2.geno --version 2
```

### Diff (compare versions)

```bash
python lmdb_matrix.py diff BXD ./lmdb_store --from 1 --to 2
```

Shows changed cells with marker and sample names:
```
Differences between v1 and v2:
  Total changed cells: 3 / 1449558 (0.0001%)
  Changed markers (rows): 3
  Changed samples (cols): 3

Marker               Sample     v1     v2
--------------------------------------------------
rs30514876           BXD9       0      2
rs13476004           BXD21      0      3
...
```

### Stats (storage analysis)

```bash
python lmdb_matrix.py stats BXD ./lmdb_store
```

Shows version count, delta efficiency, per-version sizes, and compression info (if enabled):
```
Dataset: BXD
--------------------------------------------------
Total versions:       12
Full snapshots:       2
Deltas:               10

Total payload:        5662.66 KB
  Full snapshots:     2831.19 KB
  Deltas:             2831.48 KB

Without deltas:       16987.12 KB
Storage savings:      66.7%
--------------------------------------------------
```

---

## What You Get

Reconstructing returns a `GenotypeMatrix`:

| Attribute | Description |
|-----------|-------------|
| `matrix` | `numpy.ndarray` — genotype data as numeric codes (markers × samples) |
| `markers` | Marker/SNP names |
| `samples` | Strain or sample IDs |
| `chromosomes` | Chromosome per marker |
| `cM` | Genetic positions (centiMorgans) per marker |
| `Mb` | Physical positions (megabases) per marker |
| `allele_map` | Symbol → code mapping (e.g., `{'B': 0, 'D': 1}`) |
| `founders` | Parental strains |
| `cross_type` | Cross type (riset, hs, f2, etc.) |

**Numeric codes:**
- Founder alleles: `0`, `1`, `2`, ... (up to 8 founders)
- Heterozygous: `het_code`
- Unknown/missing: `unk_code`

Decode to symbols:
```python
genotype = store.get_matrix('BXD', version=2)
symbols = genotype.decode_row(0)  # ['B', 'B', 'D', 'D', ...]
```

---

## Compression

Compression is **opt-in** and **set at store creation time** via `--compression zlib` on `import-genotype`.

- Once set, the compression algorithm is stored in LMDB and applies to **all datasets in that store**.
- `update-genotype` reads the store's compression config automatically — you do **not** pass `--compression` again.
- Mixed compression within one store is rejected. If a store was created with zlib, all subsequent imports and updates use zlib.
- Without `--compression`, payloads are stored uncompressed.

The hash chain is computed over the **compressed bytes**, so verification never needs to decompress.

## Storage Behavior

| Scenario | Storage Type | Metadata Source |
|----------|-------------|-----------------|
| Import (v1) | Full snapshot | From `.geno` file |
| Small changes | Sparse delta | Reused from last full snapshot |
| Large changes | XOR delta | Reused from last full snapshot |
| Shape change | Full snapshot | **From new `.geno` file** |
| Every 10 versions | Full snapshot | **From new `.geno` file** |

---

## Tests

```bash
guix shell python-wrapper python-lmdb python-numpy python-pytest -- \
  python -m pytest tests/ -v
```

---

## Examples

See `examples/`:
- `metadata_preservation.py`
- `cli_update_workflow.py`
- `parser_comment_fix.py`
