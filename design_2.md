# Provably Verifiable Genotype Storage (LMDB)

## 1. Overview

This document specifies an **LMDB-based, cryptographically verifiable genotype storage system** that provides:

* A **single, computation-agnostic genotype matrix** (markers × samples)
* **Append-only, hash-chained provenance** for matrix data
* **Exact historical reproducibility** of past matrix states
* **Post-hoc tamper detection** (provably verifiable state)

The storage system owns the canonical genotype matrix and its physical metadata (markers, samples, chromosomes, positions, allele encoding). Tool-specific semantic interpretation (e.g. r/qtl encoding, GEMMA filters) is the responsibility of downstream analysis pipelines, which may optionally implement external provenance tracking.

The system avoids duplication of large genotype matrices via delta encoding while providing audit-grade data integrity.

---

## 2. Goals

### 2.1 Functional Goals

* Store large genotype matrices efficiently
* Allow genotype matrices to evolve over time
* Support multiple analysis frameworks by serving canonical matrices
* Enable exact reproduction of any historical matrix state

### 2.2 Non-Functional Goals

* Detect silent corruption or tampering
* Provide cryptographic verification of historical states
* Minimize storage overhead
* Integrate cleanly with LMDB and downstream tools (e.g. r/qtl2)

---

## 3. Non-Goals

* Per-marker or per-cell provenance
* Distributed consensus or blockchain
* Authentication or authorization
* Real-time multi-writer updates

---

## 4. Core Concepts

### 4.1 Physical Data vs Semantic Interpretation

The system enforces a strict separation between physical data and interpretation:

| Layer            | Description                               | Owner                 |
| ---------------- | ----------------------------------------- | --------------------- |
| Genotype Matrix  | Physical measurement (markers × samples)  | Storage system        |
| Physical Metadata| Markers, samples, chromosomes, positions  | Storage system        |
| Semantic Metadata| Tool-specific parameters (encoding, etc.) | Analysis pipeline     |
| Analysis Results | Derived outputs                           | Out of scope          |

**Invariant**

> The genotype matrix is computation-agnostic.
> The storage system guarantees matrix integrity.
> Tool-specific interpretation is external to the storage layer.

---

## 5. Data Model (LMDB Tables)

All writes that advance dataset state **MUST occur in a single LMDB write transaction**.

---

### 5.1 `genotypes` — Current Matrix Snapshot (Cache)

Stores the **latest reconstructed matrix snapshot** for fast access.

**Key**

```text
<dataset_id>
```

**Value (binary-encoded)**

```json
{
  "matrix_version": 12,
  "matrix_hash": "H12",
  "prev_matrix_hash": "H11",
  "matrix_blob": "<binary matrix snapshot>"
}
```

**Properties**

* Overwritten on update
* Fast access for current analyses
* Not authoritative for history or verification

---

### 5.2 `matrix_history` — Immutable Matrix Ledger (Authoritative)

Append-only ledger of matrix versions.

**Key**

```text
<dataset_id>:<zero-padded matrix_version>
```

**Value (canonical binary encoding)**

```json
{
  "dataset_id": "ds1",
  "matrix_version": 12,
  "matrix_hash": "H12",
  "prev_matrix_hash": "H11",

  "storage_type": "delta",        // "full" or "delta"
  "payload": "<binary snapshot or delta>",

  "timestamp": "2025-07-25T14:30:00Z",
  "reason": "QC correction after resequencing",
  "author": "pipeline_v3"
}
```

**Properties**

* Append-only
* Hash-chained
* Sole authoritative source for matrix history

---

### 5.3 Matrix Reconstruction Rules (Normative)

To reconstruct matrix version `N`:

1. Locate the **nearest previous full snapshot** `K ≤ N`
2. Load snapshot `K`
3. Sequentially apply deltas `K+1 … N`
4. Recompute all hashes during reconstruction

**Rules**

* Every dataset MUST include:

  * A full snapshot at version 1
  * A full snapshot at least every *M* versions (implementation-defined)
* Deltas MUST be:

  * Deterministic
  * Invertible or replayable
* Reconstruction MUST NOT rely on external state

**Failure Mode**

> If any delta or snapshot is missing or corrupt, reconstruction fails and the dataset is invalid.

---

### 5.4 `meta_versions` — Immutable Semantic Metadata (Deferred)

> **Status:** Deferred from core storage system. Tool-specific metadata belongs to analysis pipelines. This section documents the original design for optional downstream implementation.

Stores tool-specific metadata bound cryptographically to a matrix state. Analysis pipelines may implement this externally using `compute_metadata_hash()` from the storage library.

**Key**

```text
<dataset_id>:<tool>:<meta_hash>
```

**Value (canonical binary encoding)**

Example: r/qtl

```json
{
  "dataset_id": "ds1",
  "tool": "rqtl",
  "matrix_hash": "H12",

  "encoding": "AA/AB/BB",
  "missing": "NA",

  "marker_order_hash": "O_markers_1",
  "sample_order_hash": "O_samples_1",

  "cross_type": "f2",
  "timestamp": "2025-07-25T14:35:00Z"
}
```

Example: GEMMA

```json
{
  "dataset_id": "ds1",
  "tool": "gemma",
  "matrix_hash": "H12",

  "encoding": "0/1/2",
  "maf_filter": 0.05,
  "center": true,

  "timestamp": "2025-07-25T14:36:00Z"
}
```

**Properties**

* Immutable
* Reusable if identical
* Semantically bound to a specific matrix hash
* **Not stored in core LMDB — managed by analysis pipeline if needed**

---

### 5.5 `meta_history` — Metadata Ledger (Deferred)

> **Status:** Deferred from core storage system.

Tracks metadata evolution per tool. May be implemented externally by analysis pipelines that require unified audit trails.

**Key**

```text
<dataset_id>:<tool>:<zero-padded meta_version>
```

**Value**

```json
{
  "dataset_id": "ds1",
  "tool": "rqtl",
  "meta_version": 3,
  "meta_hash": "M_rqtl_3",
  "prev_meta_hash": "M_rqtl_2",
  "matrix_hash": "H12",

  "timestamp": "2025-07-25T14:35:00Z",
  "reason": "Changed missing value encoding",
  "author": "analyst_1"
}
```

**Properties**

* Append-only
* Hash-chained
* Detects metadata tampering
* **Not stored in core LMDB — managed by analysis pipeline if needed**

---

### 5.6 `info` — Dataset HEAD (Non-Authoritative)

Tracks the current pointers only.

**Key**

```text
<dataset_id>
```

**Value**

```json
{
  "description": "BXD genotype release",
  "author": "GeneNetwork",

  "current_matrix_version": 12,
  "current_matrix_hash": "H12"
}
```

**Invariant**

> `info` is a cache. Validity is determined solely by ledger verification.

---

## 6. Canonical Serialization (Critical)

All hashes operate over **canonical binary encodings**.

### 6.1 Requirements

* Deterministic serialization
* Sorted keys
* UTF-8
* Explicit field inclusion list
* No whitespace or formatting variance

**Recommended formats**

* CBOR (canonical mode)
* Protobuf (deterministic serialization)
* FlatBuffers

---

## 7. Hashing Rules (Domain-Separated)

### 7.1 Matrix Hash

```text
Hn = SHA256(
  "MATRIX_V1" ||
  payload_binary ||
  prev_matrix_hash
)
```

---

### 7.2 Metadata Hash

```text
M = SHA256(
  "META_V1" ||
  tool ||
  matrix_hash ||
  canonical_semantic_fields
)
```

---

## 8. Verification Model (Tamper Detection)

### 8.1 Matrix Verification

1. Load full snapshot at version 1
2. Replay deltas sequentially
3. Recompute hashes
4. Compare against stored `matrix_hash`

---

### 8.2 Metadata Verification (Optional)

Analysis pipelines that implement external semantic metadata tracking may verify:

1. Walk pipeline's `meta_history`
2. Recompute each `meta_hash`
3. Verify matrix hash binding (matrix must exist in storage)
4. Reject orphaned metadata

The storage system provides `compute_metadata_hash()` for this purpose but does not manage semantic metadata directly.

---

## 9. Example: Matrix Reconstruction

**Goal**

Reconstruct matrix version 9.

**Ledger**

```text
v1  (full)
v2  (delta)
v3  (delta)
v4  (full)
v5  (delta)
v6  (delta)
v7  (delta)
v8  (delta)
v9  (delta)
```

**Steps**

1. Load full snapshot v4
2. Apply deltas v5 → v9
3. Recompute H5…H9
4. Confirm H9 matches ledger

---

## 10. Example: Reproducing a Historical Analysis

```text
Dataset: ds1
Matrix version: 9
```

**Procedure (Storage System)**

1. Reconstruct matrix v9 from `matrix_history`
2. Verify matrix hash chain (H1 → H2 → ... → H9)
3. Return canonical genotype matrix with physical metadata

**Procedure (Analysis Pipeline)**

4. Load tool-specific parameters from pipeline's provenance record
5. Transform canonical matrix for target tool (r/qtl2, GEMMA, etc.)
6. Run analysis
7. Output is reproducible because both matrix and tool params are versioned

---

## 11. Invariants (Enforced)

1. Matrix history is append-only
2. Every matrix hash is reachable from history
3. HEAD pointers must refer to existing hashes
4. Hash mismatch invalidates the dataset
5. Canonical serialization is mandatory
6. All state updates occur in a single LMDB transaction

**Optional invariants** (for pipelines implementing semantic metadata tracking):

7. Metadata history is append-only (external to storage system)
8. Every metadata hash binds to a valid matrix hash (external to storage system)

---

## 12. Threat Model Clarification

This system provides:

* **Post-hoc tamper detection**
* **Cryptographic auditability**
* **Exact reproducibility**

It does **not** prevent a malicious writer from altering history, but such changes are **detectable** by any verifier with read access.

---

## 13. Summary

This design defines a **Git-like, LMDB-native, cryptographically verifiable genotype storage system** that:

* Stores genotype data once with delta encoding
* Guarantees matrix integrity via hash-chained provenance
* Enables exact reproduction of any historical matrix state
* Detects corruption or tampering
* Scales cleanly to large datasets

The storage system owns the canonical matrix and its physical metadata. Tool-specific semantic interpretation is the responsibility of downstream analysis pipelines, which may optionally implement external provenance tracking using `compute_metadata_hash()`.

It is suitable for production pipelines, long-term archives, and publication-quality computational genetics.

---

## 14. TODO / Future Improvements

| # | Improvement | Status | Priority |
|---|-------------|--------|----------|
| 1 | **Remove/fix `geno_storage/cli.py`** — Dead/outdated file that still passes `.matrix` instead of full `GenotypeMatrix`, breaking metadata preservation. | ✅ **DONE** | High |
| 2 | **Add `diff` command** — Show differences between two versions without manual reconstruction. | ✅ **DONE** | High |
| 3 | **Add `stats` command** — Show storage efficiency, version counts, delta sizes. | ✅ **DONE** | Medium |
| 4 | **Add `.geno` export** — Round-trip: LMDB → `.geno` file format. | ✅ **DONE** | Medium |
| 5 | **Better error messages** — Friendly messages for missing versions, corrupted chains, shape mismatches. | ✅ **DONE** | Medium |
| 6 | **`setup.py` entry points** — Add `console_scripts` for pip install. | ⏳ **NOT DONE** | Low |
| 7 | **Fast verify option** — Check hash chain linkage without full payload recomputation. | ✅ **DONE** | Low |
| 8 | **`__main__.py`** — Support `python -m geno_storage`. | ⏳ **NOT DONE** | Low |
| 9 | **Per-dataset `FULL_SNAPSHOT_INTERVAL`** — Make interval configurable per dataset. | ⏳ **NOT DONE** | Low |
