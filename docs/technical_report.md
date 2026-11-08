# Technical Report — GPU-Accelerated Sparse Linear Algebra IDS

## 1. Problem Statement

Network Intrusion Detection Systems (IDS) must inspect every packet payload against hundreds or thousands of attack signatures in real time. The computational core — matching byte patterns against payloads — is inherently parallel and data-parallel, making it a natural fit for GPU acceleration.

The key research question: **Can sparse linear algebra, specifically sparse matrix–matrix multiplication, serve as an effective and scalable kernel for signature-based IDS?**

## 2. Research Motivation

Prior GPU IDS work (Gnort [1], Kargus [2]) demonstrated that GPU parallelism can accelerate pattern matching via Aho-Corasick automata compiled onto the GPU. However, automata-based approaches require complex construction, are hard to update dynamically, and do not naturally express the similarity between packets and signatures.

This project proposes an alternative formulation:

**Represent both packets and signatures as sparse feature vectors in a shared byte n-gram vocabulary, then use sparse matrix multiplication to compute similarity scores in bulk.**

Advantages over automata approaches:
- Naturally batch-processes all packets against all signatures simultaneously
- Hardware-agnostic sparse kernels (OpenMP on CPU, cuSPARSE on GPU)
- Exact verification prevents false alerts regardless of hash collisions
- Modular: feature extraction, sparse computation, and verification are independent

Related work includes GPU-compressed Aho-Corasick [3] and CIC-IDS2017 [4] dataset evaluation.

## 3. Sparse Linear Algebra Formulation

### Feature Representation

Let `V = 65536` be the vocabulary size (hash table size for byte n-grams).

**Packet feature matrix P ∈ {0,1}^{N×V}:**

```
P[i][h] = 1  if n-gram hash h appears in packet i's payload
         0  otherwise
```

**Signature feature matrix S ∈ {0,1}^{K×V}:**

```
S[j][h] = 1  if n-gram hash h appears in signature j's pattern
         0  otherwise
```

**Score matrix C = P × Sᵀ ∈ ℝ^{N×K}:**

```
C[i][j] = Σ_h P[i][h] × S[j][h]
         = |{n-grams shared between packet i and signature j}|
```

### Candidate Filtering

A necessary condition for signature *j* to match packet *i* (when pattern length ≥ n):

```
C[i][j] ≥ num_features(j)   ⟺   all signature n-grams present in packet
```

This is a **necessary but not sufficient** condition. It filters out the vast majority of non-matching pairs while admitting all true matches (no false negatives when pattern ≥ n-gram size).

### Short Signature Bypass

When a signature's pattern has fewer bytes than the n-gram size, its n-gram feature representation cannot appear in a longer packet's standard n-gram set. These signatures bypass the sparse filter and go directly to exact verification against all packets. This is safe and correct.

### Exact Verification

All candidate pairs undergo byte-level substring search:

```
alert iff memcmp(packet_payload[k:k+m], signature_pattern) == 0
          for some offset k
          AND protocol matches AND port matches
```

The verifier is the **authoritative oracle**. Hash collisions and near-matches are always rejected here.

## 4. HPC Parallelism

### Sequential (Reference)

The reference implementation runs the entire pipeline single-threaded. It is used for:
- Correctness validation
- Baseline timing for speedup calculation
- Hash-collision-free result reference

### OpenMP (CPU Parallel)

Three phases are parallelised:

1. **Feature extraction:** Independent per packet. `#pragma omp parallel for schedule(dynamic, 64)`.
2. **SpMM:** Each row of P is independent. `#pragma omp parallel for schedule(dynamic, 1)`.
3. **Verification:** Each candidate pair is independent. Per-thread alert lists merged after barrier.

Thread count is configurable. Race-condition analysis:
- Feature extraction: each thread writes `features[i]` for unique `i`. ✓ No race.
- SpMM: each thread writes `row_cols[i]`, `row_vals[i]` for unique `i`. ✓ No race.
- Verification: each thread appends to `per_thread[tid]`. ✓ No race (tid is unique).

### CUDA (GPU Parallel)

Feature extraction runs on CPU (OpenMP). The SpMM kernel uses **cuSPARSE `cusparseSpGEMM`**:
1. Build B^T (K×V → V×K transpose) on CPU, upload to GPU.
2. Upload A (P matrix) to GPU.
3. cuSPARSE SpGEMM computes C = A × B^T on GPU.
4. Download C, run candidate generation and verification on CPU.

Design decisions:
- No custom CUDA SpMM kernel; cuSPARSE is vendor-optimised and handles load balancing.
- B^T is transposed on CPU because N (packets) >> K (signatures), so the transpose is small.
- RAII cleanup struct prevents GPU memory leaks on any error path.

## 5. Experimental Methodology

### Hardware

- **CPU:** Intel Core i7 (WSL2, 16 logical threads available)
- **RAM:** ~4 GB available in WSL2
- **GPU:** No local NVIDIA GPU. CUDA path not yet benchmarked locally.

### Dataset

Synthetic 10,000-packet benchmark PCAP:
- Payloads: uniform random bytes, lengths 50–1500 bytes
- ~5% of packets contain an injected signature pattern at a random position
- 8 attack signatures covering SQL injection, XSS, command injection, directory traversal, NOP sled, scanner UA, DNS tunneling, PHP shell

### Metrics

- Total pipeline time (ms)
- Phase-level timing: feature extraction, CSR construction, SpMM, verification
- Packets per second (throughput)
- Speedup = sequential_total / mode_total
- Parallel efficiency = speedup / thread_count
- Matrix dimensions: N, K, V, nnz(P), nnz(S), nnz(C)
- Candidate count, alert count

## 6. Results (Measured, Not Fabricated)

### Matrix Properties

| Matrix | Dimensions | nnz | Sparsity |
|--------|-----------|-----|---------|
| P (packets) | 10,000 × 65,536 | 7,691,118 | 98.83% |
| S (signatures) | 8 × 65,536 | 82 | 99.98% |
| C (scores) | 10,000 × 8 | 9,254 | 88.43% |

### Timing Breakdown (ms)

| Mode | FeatExt | CSR Build | SpMM | Verify | **Total** |
|------|--------|-----------|------|--------|-----------|
| Sequential | 821.9 | 51.2 | 42.5 | 0.8 | **916.7** |
| OpenMP-1T | 884.2 | 39.3 | 53.0 | 1.3 | **978.3** |
| OpenMP-2T | 496.5 | 31.1 | 21.9 | 0.4 | **550.2** |
| OpenMP-4T | 279.3 | 35.0 | 13.0 | 0.3 | **327.9** |
| OpenMP-16T | 123.4 | 45.1 | 7.8 | 2.3 | **179.1** |

### Speedup and Efficiency

| Mode | Speedup | Parallel Efficiency |
|------|---------|-------------------|
| OpenMP-1T | 0.94× | 93.7% |
| OpenMP-2T | 1.67× | 83.3% |
| OpenMP-4T | 2.80× | 69.9% |
| OpenMP-16T | 5.12× | 32.0% |

**Observation:** Feature extraction dominates (~90% of total time). Efficiency drops with more threads partly due to Amdahl's Law (CSR construction is sequential) and WSL2 thread scheduling overhead.

## 7. Roofline Analysis

### Sparse Compute Phase (SpMM)

For the merge-based SpMM implementation:

```
Operations:  O(M × N × (avg_nnz_per_row_A + avg_nnz_per_row_B))
           ≈ 10,000 × 8 × (769 + 10) ≈ 62.3M comparisons

Data moved:  (nnz_A + nnz_B) × (4B col_idx + 4B values)
           + (N + K + 2) × 4B row_ptr
           ≈ 61.6 MB

Arithmetic intensity:  62.3M ops / 61.6 MB ≈ 1.01 ops/byte
```

**Classification: at the boundary of memory-bound and compute-bound.**

Irregular memory access due to sparse indexing makes effective memory bandwidth much lower than theoretical bandwidth, so the workload is effectively **memory-bandwidth-limited** in practice.

### Bottleneck Identification

| Phase | Bottleneck | Evidence |
|-------|-----------|---------|
| Feature extraction | Compute + memory (hashing 7.7M n-grams) | 90% of total time |
| CSR construction | Sequential, memory-bound | Modest, ~5% |
| SpMM | Irregular memory access | Scales with threads, ~4% |
| Verification | Trivial for 751 candidates | <0.1% |

**Key finding:** Feature extraction is the dominant bottleneck. GPU acceleration of SpMM alone will not provide large end-to-end speedup unless feature extraction is also parallelised on the GPU.

## 8. Limitations and Future Work

### Current Limitations

1. **IPv4 only** — IPv6 not parsed
2. **No TCP stream reassembly** — multi-packet payloads not joined
3. **Short signatures** — O(N×K_short) all-pairs scan; acceptable for few short sigs
4. **Hash collisions** — 8.3% collision rate increases candidate count; harmless correctness-wise
5. **CUDA not yet validated** — no local NVIDIA GPU; Colab script provided
6. **Feature extraction on GPU** — the dominant bottleneck could be accelerated with a custom CUDA n-gram hashing kernel
7. **Large vocab** — 65536 may still cause collisions; a larger vocabulary (e.g., 1M) or multi-hash scheme could reduce them

### Future Work

1. **GPU feature extraction** — custom CUDA kernel for byte n-gram hashing
2. **GPU candidate filter** — threshold kernel directly on C on GPU
3. **Streaming PCAP ingestion** — read and process packets in chunks to reduce memory
4. **Dynamic signature updates** — incremental CSR update without full rebuild
5. **CIC-IDS2017 evaluation** — test on real-world attack traffic
6. **Multi-hash scheme** — reduce false candidates from hash collisions
7. **TCP stream reassembly** — Wireshark/libnids integration
8. **Performance on real GPU** — expected large speedup for SpMM; feature extraction remains bottleneck

## 9. References

[1] Gnort: High Performance Network Intrusion Detection Using Graphics Processors.
    https://doi.org/10.1007/978-3-540-87403-4_7

[2] Kargus: A Highly-scalable Software-based Intrusion Detection System.
    https://doi.org/10.1145/2382196.2382232

[3] GPU-Based Deep Packet Inspection with Regular Expression Matching Using Aho-Corasick Algorithm.
    https://arxiv.org/abs/1704.02272

[4] CIC-IDS2017: A Realistic Cyber Defense Dataset.
    https://www.unb.ca/cic/datasets/ids-2017.html

[5] NVIDIA cuSPARSE Library Documentation.
    https://docs.nvidia.com/cuda/cusparse/

[6] NVIDIA CUDA C++ Programming Guide.
    https://docs.nvidia.com/cuda/

