# Implementation Workflow Checklist

**Provably Verifiable Genotype Storage (LMDB)**

---

## 0. Foundations (Do First)

* ☐ Choose **canonical binary format** (CBOR / Protobuf / FlatBuffers)
* ☐ Define **domain-separated hash prefixes** (`MATRIX_V1`, `META_V1`)
* ☐ Define **genotype matrix binary layout**
* ☐ Define **delta encoding format**
* ☐ Define **hash verification helpers**
* ☐ Decide **full snapshot frequency** (e.g. every 10 versions)

---

## 1. LMDB Setup

* ☐ Create LMDB environment
* ☐ Create DBIs:

  * ☐ `genotypes` (current snapshot)
  * ☐ `matrix_history` (append-only)
  * ☐ `meta_versions`
  * ☐ `meta_history`
  * ☐ `info` (HEAD)
* ☐ Enforce **single writer transaction model**

---

## 2. Matrix Versioning

### 2.1 Initial Import

* ☐ Encode matrix → canonical binary
* ☐ Store **v1 full snapshot** in `matrix_history`
* ☐ Compute `H1`
* ☐ Write `genotypes` cache
* ☐ Initialize `info` HEAD

---

### 2.2 Update Matrix (vN → vN+1)

* ☐ Load current matrix (from cache or reconstruct)
* ☐ Compute delta vs previous version
* ☐ If delta too large → store full snapshot
* ☐ Compute new matrix hash
* ☐ Append to `matrix_history`
* ☐ Update `genotypes` cache
* ☐ Update `info` HEAD
* ☐ Commit LMDB transaction

---

## 3. Matrix Reconstruction

* ☐ Locate target version in `matrix_history`
* ☐ Find nearest previous full snapshot
* ☐ Load full snapshot
* ☐ Replay deltas forward
* ☐ Recompute and verify hashes
* ☐ Return reconstructed matrix

---

## 4. Metadata Versioning (Per Tool)

### 4.1 Add Metadata

* ☐ Define semantic fields (tool-specific)
* ☐ Canonically serialize metadata
* ☐ Compute `meta_hash` (bind to `matrix_hash`)
* ☐ Store in `meta_versions`
* ☐ Append to `meta_history`
* ☐ Update `info.current_meta[tool]`

---

### 4.2 Reuse Metadata

* ☐ Check if identical `meta_hash` exists
* ☐ Reuse existing entry if present

---

## 5. Verification

### 5.1 Matrix Verification

* ☐ Walk `matrix_history` from v1 → vN
* ☐ Recompute all hashes
* ☐ Detect any mismatch

---

### 5.2 Metadata Verification

* ☐ Walk `meta_history` per tool
* ☐ Recompute metadata hashes
* ☐ Verify matrix hash bindings
* ☐ Reject orphaned metadata

---

## 6. Reproduce Historical Analysis

* ☐ Reconstruct matrix version V
* ☐ Load metadata by `meta_hash`
* ☐ Verify both hash chains
* ☐ Run analysis
* ☐ Record results externally (out of scope)

---

## 7. Invariant Enforcement

* ☐ History tables are append-only
* ☐ No metadata without valid matrix hash
* ☐ HEAD is never authoritative
* ☐ All updates occur in one transaction
* ☐ Hash mismatch = invalid dataset

---

## 8. Optional Enhancements

* ☐ Digital signatures on hashes
* ☐ External checkpoint hash (Git tag / DOI)
* ☐ Inverse deltas for faster rollback
* ☐ Performance benchmarks
* ☐ R / C / Python bindings

---

### Final Mental Model

> **History = truth**
> **HEAD = convenience**
> **Hashes = law**
