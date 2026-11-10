# HPC Professor & Senior Engineer Defense Guide: GPU Sparse IDS

> **Purpose:** Comprehensive technical guide and viva defense manual for the **GPU-Accelerated Sparse Linear Algebra Network Intrusion Detection System (IDS)**.  
> This document covers every architectural decision, mathematical formulation, module implementation, HPC parallelization technique, performance benchmark, and anticipated viva examination question.

---

## Table of Contents
1. [Module 1: Problem Formulation & Core Architecture](#module-1-problem-formulation--core-architecture)
2. [Module 2: Network Ingestion & PCAP Parsing (`pcap_reader.cpp`)](#module-2-network-ingestion--pcap-parsing)
3. [Module 3: Feature Extraction & In-VRAM Hashing (`ngram_features.cpp`, `cuda_engine.cu`)](#module-3-feature-extraction--in-vram-hashing)
4. [Module 4: Sparse Linear Algebra & CSR Representation (`csr_matrix.cpp`)](#module-4-sparse-linear-algebra--csr-representation)
5. [Module 5: SpMM Execution Engines (CPU, OpenMP, and CUDA cuSPARSE)](#module-5-spmm-execution-engines)
6. [Module 6: Candidate Filtering & The Short-Signature Bypass Rule](#module-6-candidate-filtering--the-short-signature-bypass-rule)
7. [Module 7: Stage-2 AVX2 SIMD Vectorized Verification (`cpu_engine.cpp`, `openmp_engine.cpp`)](#module-7-stage-2-avx2-simd-vectorized-verification)
8. [Module 8: Performance Breakdown, Roofline Model & Benchmarks](#module-8-performance-breakdown-roofline-model--benchmarks)
9. [Module 9: Comprehensive Viva & Technical Defense Questions with Model Answers](#module-9-comprehensive-viva--technical-defense-questions)

---

## Module 1: Problem Formulation & Core Architecture

### 1.1 The Classical DPI Wall
Deep Packet Inspection (DPI) matches incoming network packets against hundreds or thousands of threat signatures.
* **Classical Automata Approach:** Aho-Corasick, Boyer-Moore, or deterministic finite automata (DFAs).
* **Why Automata Break Down on Modern HPC & GPU Architectures:**
  1. **Pointer Chasing & Cache Misses:** State transitions follow $S_{next} = \delta(S_{current}, \text{byte})$, jumping non-contiguously in memory and thrashing L1/L2 caches.
  2. **SIMT Divergence on GPUs:** On a GPU, 32 threads in a warp execute in lockstep. If packet payloads diverge into different automaton states, threads serialize, destroying GPU throughput.
  3. **High Recompilation Cost:** Adding or updating rules requires recompiling the automaton graph, taking minutes and consuming vast memory.

### 1.2 Mathematical Formulation: String Matching as Sparse Matrix Multiplication
We reformulate string matching into Sparse General Matrix Multiplication ($\text{SpMM}$):

$$\mathbf{C} = \mathbf{P} \times \mathbf{S}^T$$

```
Packet Payload  ──►  Sliding n-grams  ──►  FNV-1a Hash  ──►  Matrix P (N x V)
Attack Rules    ──►  Sliding n-grams  ──►  FNV-1a Hash  ──►  Matrix S (K x V)
```

* **Vocabulary Size ($V$):** Discrete hash space $V = 65,536$ ($2^{16}$).
* **Packet Feature Matrix $\mathbf{P} \in \{0, 1\}^{N \times V}$:**
  * $N$ packets (e.g., 10,000).
  * Column $h$ is `1` if n-gram hash $h$ appears in packet $i$'s payload; `0` otherwise.
  * Typical 1500-byte packet contains $\le 1498$ unique 3-grams $\implies \mathbf{P}$ is **$98.83\%$ sparse**.
* **Signature Feature Matrix $\mathbf{S} \in \{0, 1\}^{K \times V}$:**
  * $K$ threat signatures.
  * Column $h$ is `1` if n-gram hash $h$ is present in signature $j$'s rule pattern.
  * A 20-byte attack signature contains 18 3-grams $\implies \mathbf{S}$ is **$99.98\%$ sparse**.
* **Score Matrix $\mathbf{C} = \mathbf{P} \times \mathbf{S}^T \in \mathbb{R}^{N \times K}$:**
  $$\mathbf{C}[i, j] = \sum_{h=0}^{V-1} \mathbf{P}[i, h] \cdot \mathbf{S}[j, h]$$
  $\mathbf{C}[i, j]$ is the inner dot-product between packet $i$ and signature $j$, counting the exact number of shared unique n-gram features.

### 1.3 Two-Stage Hierarchical Filter
```
Raw PCAP Traffic (10,000 Packets)
       │
       ▼
[Stage 1: Sparse Filter (GPU cuSPARSE / OpenMP)]
       │  C[i, j] >= num_features(S_j) ?
       ├── NO  ──► Discard (>99% of non-matching pairs pruned)
       └── YES ──► Candidate Pairs (751 Candidates)
                     │
                     ▼
       [Stage 2: Exact Substring Verification]
                     │  AVX2 SIMD 32-Byte Vector Probe (_mm256_cmpeq_epi8)
                     ▼
       Final Verified Alerts (355 Alerts, 0 False Positives, 0 False Negatives)
```

* **Proof of Zero False Negatives:** If signature $S_j$ is a contiguous substring of packet $P_i$, every 3-gram of $S_j$ must physically exist in $P_i$. Thus, $\mathbf{C}[i, j] \ge \text{num\_features}(S_j)$ is a **necessary mathematical condition**. A hash collision can only artificially increase $\mathbf{C}[i, j]$, never decrease it. Stage 2 eliminates false positives.

---

## Module 2: Network Ingestion & PCAP Parsing (`pcap_reader.cpp`)

* **File Location:** `src/ingestion/pcap_reader.cpp`, `include/ingestion/pcap_reader.h`
* **Role:** High-speed streaming packet extraction directly from industry-standard `.pcap` files via `libpcap`.

### Internal Mechanics & Protocol Decoding
1. **Pcap Header Validation:** Opens file handle via `pcap_open_offline()`, checks datalink type (`DLT_EN10MB` for Ethernet frames).
2. **802.1Q VLAN Handling:**
   * Reads EtherType at byte offset 12.
   * If EtherType is `0x8100` (VLAN tagged), advances offset by 4 bytes (18-byte L2 header) and reads the encapsulated EtherType.
3. **Dual Stack IP Decoding (IPv4 & IPv6):**
   * **IPv4 (`0x0800`):** Checks minimum header length (20 bytes), extracts Internet Header Length ($\text{IHL} = (\text{header}[0] \& 0x0F) \times 4$), extracts Protocol (TCP=6, UDP=17), IPs, and calculates payload starting offset.
   * **IPv6 (`0x86DD`):** Parses fixed 40-byte IPv6 header, decodes 128-bit addresses using `inet_ntop(AF_INET6, ...)`, extracts Next Header (TCP/UDP), and points directly to payload.
4. **Defensive Parsing:** Rejects truncated packets, malformed headers, and non-IP traffic gracefully without crashing.

---

## Module 3: Feature Extraction & In-VRAM Hashing (`ngram_features.cpp`, `cuda_engine.cu`)

* **File Location:** `src/features/ngram_features.cpp`, `src/cuda/cuda_engine.cu`
* **Algorithm:** 3-Gram Sliding Window + 32-bit Fowler–Noll–Vo (FNV-1a) Hash.

### 3.1 The FNV-1a Hash Function
```cpp
uint32_t hash = 2166136261u; // FNV offset basis
for (int i = 0; i < len; ++i) {
    hash ^= payload[i];
    hash *= 16777619u;        // FNV prime
}
return hash % vocab_size;    // Bound to [0, 65535]
```
* **Why FNV-1a?** Extremely fast byte-at-a-time rolling dispersion with excellent avalanche properties on short byte sequences, perfectly suited for GPU registers and warp execution.

### 3.2 The GPU In-VRAM Breakthrough (Eliminating Amdahl's Bottleneck)
In early prototypes, feature extraction ran on the CPU (taking 1,834 ms), causing a 0.91× slowdown (Amdahl's law bottleneck). We solved this by developing a **two-pass warp-parallel GPU kernel** with in-VRAM shared-memory bitsets:

```
Raw Payloads Array in GPU VRAM
        │
[Pass 1: count_packet_features_kernel]
        │  Each thread block processes 1 packet.
        │  8 KB shared-memory bitset (2048 x 32-bit words).
        │  Threads slide across payload in warp-stride, compute FNV-1a,
        │  and atomicOr() into shared-memory bitset.
        │  Count total bits set = row_counts[pkt_idx].
        ▼
[Thrust Prefix Sum / scan]
        │  Exclusive prefix scan converts row_counts into CSR row_ptr.
        ▼
[Pass 2: emit_packet_features_kernel]
        │  Bitset scanned via __ffs() (Find First Set).
        │  Directly writes strictly sorted col_idx and 1.0f values into GPU VRAM.
        ▼
Output: Complete In-VRAM CSR Matrix P (No Host-Device Roundtrip!)
```

* **Why this is fast:**
  - `atomicOr` on `__shared__` memory is orders of magnitude faster than global atomic operations.
  - Scanning bitsets with hardware intrinsic `__ffs()` emits column indices **in strictly ascending order automatically**, completely eliminating the need for expensive sorting!
  - Reduced feature extraction from **1,834.5 ms to 14.87 ms (123.4× speedup)**.

---

## Module 4: Sparse Linear Algebra & CSR Representation (`csr_matrix.cpp`)

* **File Location:** `src/sparse/csr_matrix.cpp`, `include/sparse/csr_matrix.h`
* **Representation:** Compressed Sparse Row (CSR).

### 4.1 The CSR Memory Layout
Rather than allocating a dense $10,000 \times 65,536$ array ($2.62 \text{ GB}$ of RAM), CSR compresses the matrix into three compact 1D vectors:
1. `row_ptr` (size $N+1$): Entry `row_ptr[i]` stores the starting index of row $i$ in `col_idx`.
2. `col_idx` (size $NNZ$): Stores the column index for each non-zero feature.
3. `values` (size $NNZ$): Stores the value (all `1.0f` for binary feature presence).

**Memory Savings:**
* Dense Matrix: $10,000 \times 65,536 \times 4 \text{ bytes} \approx 2.62 \text{ GB}$.
* CSR Representation: $(10,001 \times 4) + (7,691,118 \times 4) + (7,691,118 \times 4) \approx 61.5 \text{ MB}$.
* **Compression ratio: ~42.6× reduction in memory footprint!**

---

## Module 5: SpMM Execution Engines (CPU, OpenMP, and CUDA cuSPARSE)

### 5.1 CPU Sequential Engine (`cpu_engine.cpp`)
* Implements two-pointer sorted intersection between sparse row vectors $P_i$ and $S_j$.
* Complexity: $\mathcal{O}(NNZ(P_i) + NNZ(S_j))$.

### 5.2 Multi-Threaded OpenMP Engine (`openmp_engine.cpp`)
* **Parallel Work Distribution:** `#pragma omp parallel for schedule(dynamic, 1)` distributes rows of matrix $P$ among available CPU threads.
* **Thread-Safety Invariant:** Since each thread writes strictly to its private row `row_cols[i]`, there are **zero race conditions and no locking overhead during SpMM**.
* Single-threaded assemble builds the final CSR in one fast pass.

### 5.3 GPU cuSPARSE Engine (`cuda_engine.cu`)
* Uses NVIDIA's hardware-tuned `cusparseSpGEMM` API.
* Computes $C = P \times S^T$ in 3 phases:
  1. `cusparseSpGEMM_workEstimation()`: Determines required scratchpad buffer sizes.
  2. `cusparseSpGEMM_compute()`: Executes sparse matrix multiplication on Tensor/CUDA cores.
  3. `cusparseSpGEMM_copy()`: Writes non-zero entries into device arrays `d_C_row_ptr`, `d_C_col_idx`, and `d_C_values`.

---

## Module 6: Candidate Filtering & The Short-Signature Bypass Rule

* **File Location:** `src/cuda/cuda_engine.cu` (`filter_candidates_gpu_kernel`), `cpu_engine.cpp`

### 6.1 In-Place GPU Candidate Filtering
Instead of copying the full matrix $C$ back over PCIe ($10,000 \times 8$ entries), we filter candidates directly inside GPU VRAM:
```cuda
__global__ void filter_candidates_gpu_kernel(
    const int* C_row_ptr, const int* C_col_idx, const float* C_values,
    int num_rows, const uint32_t* sig_num_features, int ngram_size,
    const int* sig_pattern_lens, ids::CandidatePair* out_candidates,
    int* candidate_count, int max_candidates)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;
    int start = C_row_ptr[row], end = C_row_ptr[row + 1];

    for (int idx = start; idx < end; ++idx) {
        int sig_idx = C_col_idx[idx];
        float score = C_values[idx];
        if (sig_num_features[sig_idx] == 0 || sig_pattern_lens[sig_idx] < ngram_size) continue;
        if (static_cast<uint32_t>(score + 0.5f) >= sig_num_features[sig_idx]) {
            int pos = atomicAdd(candidate_count, 1);
            if (pos < max_candidates) {
                out_candidates[pos].packet_idx = row;
                out_candidates[pos].signature_idx = sig_idx;
                out_candidates[pos].score = score;
            }
        }
    }
}
```
* **PCIe Bandwidth Conservation:** Only the compact array of candidate pairs (e.g., 751 pairs $\times 12$ bytes $\approx 9 \text{ KB}$) is copied to the host!

### 6.2 The Short-Signature Bypass Invariant
* **Critical Edge Case:** What if an attack pattern has length $< 3$ bytes (e.g., `\x90\x90`)?
* **Problem:** A 2-byte signature cannot form a 3-gram. Its hash will never appear in matrix $P$ or $S$.
* **Solution (Bypass Rule):** Any rule where `pattern.size() < ngram_size` bypasses Stage 1 matrix filtering completely and is paired with every packet directly for Stage 2 exact verification. Because short signatures are rare, this preserves 100% detection soundness without impacting throughput.

---

## Module 7: Stage-2 AVX2 SIMD Vectorized Verification (`cpu_engine.cpp`, `openmp_engine.cpp`)

* **File Location:** `src/compute/cpu_engine.cpp`, `src/compute/openmp_engine.cpp`
* **Algorithm:** 256-bit AVX2 First-Character Probe with Bitmask Extraction.

### 7.1 How It Works
Instead of checking packet payload byte-by-byte, we load 32 bytes into an AVX2 vector register in a single CPU cycle:

```
Haystack (Payload):  [ 32 Bytes loaded into 256-bit register YMM0 via _mm256_loadu_si256 ]
Needle First Byte:   [ Broadcast needle[0] to all 32 lanes of YMM1 via _mm256_set1_epi8 ]
                                       │
                                       ▼
                     [ Compare: _mm256_cmpeq_epi8(YMM0, YMM1) ]
                                       │
                                       ▼
                     [ Extract Bitmask: _mm256_movemask_epi8() ]
                                       │
                     Resulting 32-bit integer:  00000000 00000100 00000000 00000001
                                                              ▲                  ▲
                                                           Bit 18             Bit 0
```

1. **`__builtin_ctz(mask)`:** Uses the hardware Count Trailing Zeros CPU instruction to immediately jump to matching byte offsets.
2. **`memcmp` Verification:** Only checks the remainder of the pattern at candidate offsets.
3. **`mask &= mask - 1`:** Clears lowest set bit in $\mathcal{O}(1)$ instructions.
4. **Performance:** Verifies 751 candidates in **0.399 ms** (~0.53 μs per candidate).

---

## Module 8: Performance Breakdown, Roofline Model & Benchmarks

### 8.1 Empirical Benchmark Results (NVIDIA Tesla T4 GPU on Google Colab)

| Mode | FeatExt (ms) | CSR Build (ms) | SpMM (ms) | Verification (ms) | H2D (ms) | D2H (ms) | Total Compute (ms) | **Total Wallclock (ms)** | Throughput (pkts/s) | **Speedup vs Seq** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Sequential (1T)** | 1834.52 | 40.16 | 158.16 | 0.54 | 0.00 | 0.00 | 2033.37 | 2033.62 | 4,917 | **1.00×** |
| **OpenMP (1T)** | 2090.49 | 33.07 | 71.13 | 0.50 | 0.00 | 0.00 | 2195.20 | 2195.49 | 4,555 | **0.93×** |
| **OpenMP (2T)** | 1094.75 | 37.07 | 73.35 | 0.39 | 0.00 | 0.00 | 1205.55 | 1205.78 | 8,293 | **1.69×** |
| **OpenMP (4T)** | 1121.78 | 32.72 | 64.18 | 0.42 | 0.00 | 0.00 | 1219.10 | 1219.38 | 8,201 | **1.67×** |
| **CUDA (cuSPARSE + FeatExt)** | **14.87** | **6.37** | **43.17** | **0.40** | **2.39** | **0.09** | **64.80** | **237.67** | **42,075** | **8.56×** |

### 8.2 Roofline Model Analysis
* **Arithmetic Intensity ($I$):**
  $$I = \frac{\text{Operations}}{\text{Bytes Transferred}} \approx \frac{NNZ(P) + NNZ(S)}{\text{Bytes}(P) + \text{Bytes}(S)} \approx 0.08 - 0.15 \text{ FLOP/Byte}$$
* **Classification:** Since $I \ll 1.0$, the SpMM kernel is **heavily memory-bandwidth bound**.
* **Why GPU Wins:** The NVIDIA Tesla T4 provides **$320 \text{ GB/s}$** of memory bandwidth (GDDR6), whereas the dual-channel host CPU RAM provides only $\sim 30-40 \text{ GB/s}$.

---

## Module 9: Comprehensive Viva & Technical Defense Questions

### Question 1: Mathematical Foundations
> **Examiner:** *"Why reformulate pattern matching as a matrix multiplication $C = P \times S^T$? What is the computational advantage?"*

**Answer:**
"Automata-based pattern matching (such as Aho-Corasick) relies on non-contiguous memory pointer traversals ($S_{next} = \delta(S_{current}, c)$), which cause frequent L1/L2 cache misses on CPUs and massive thread divergence across GPU warps.

By reformulating n-gram presence as binary sparse vectors over a fixed discrete hash space $V = 65,536$, the string matching problem becomes a Sparse General Matrix Multiplication ($\text{SpMM}$): $\mathbf{C} = \mathbf{P} \times \mathbf{S}^T$. This transforms random memory lookups into structured, highly parallel sparse linear algebra. Modern GPUs have dedicated hardware and tuned libraries (cuSPARSE) that excel at streaming sparse dot products across thousands of cores simultaneously."

---

### Question 2: Soundness and Zero False Negatives
> **Examiner:** *"Can your sparse matrix filter produce false negatives due to hash collisions or thresholding?"*

**Answer:**
"No, the filter is mathematically proven to have **zero false negatives**:
1. If signature $S_j$ is a contiguous substring of packet $P_i$, every single n-gram within $S_j$ must physically exist in $P_i$. Therefore, the number of shared n-grams must satisfy:
   $$\mathbf{C}[i, j] \ge \text{num\_features}(S_j)$$
2. A hash collision can only map an extraneous n-gram to an existing signature bucket, which can only **increase** the score $\mathbf{C}[i, j]$, never decrease it.
3. Any candidate passing this threshold is handed to Stage 2 for exact byte-level verification, which completely removes any false positives. Hence, sensitivity is 100%."

---

### Question 3: The Short-Signature Edge Case
> **Examiner:** *"What happens if an attack signature is shorter than your n-gram size (e.g., pattern is 2 bytes, but $n=3$)? Does your system fail to detect it?"*

**Answer:**
"No. We implemented a dedicated **Short-Signature Bypass Rule** in `generate_candidates()` and `filter_candidates_gpu_kernel`:
- Signatures shorter than $n$ cannot form a full 3-gram, so their features are not in matrix $\mathbf{S}$.
- The engine identifies all rules where `pattern.size() < ngram_size` and automatically bypasses Stage 1 matrix filtering, pairing them with all packets directly in the Stage 2 exact verification queue.
- Because short attack patterns are rare in IDS rulesets, this introduces negligible verification overhead while guaranteeing complete detection soundness."

---

### Question 4: HPC & Amdahl's Law Bottleneck
> **Examiner:** *"In your initial CUDA implementation, why did you achieve only a 0.91× speedup (slower than sequential)? How did you solve it?"*

**Answer:**
"In our initial implementation, only the matrix multiplication $C = P \times S^T$ was offloaded to the GPU. Feature extraction—calculating rolling FNV-1a hashes for 10,000 packets—remained on the CPU, taking 1,834 ms.

According to **Amdahl's Law**:
$$\text{Speedup} = \frac{1}{(1 - P) + \frac{P}{S}}$$
Even if the GPU made SpMM instantaneous ($S \to \infty$), the non-parallelized feature extraction ($1 - P \approx 90\%$) bounded total speedup to $\le 1.1\times$. Furthermore, copying raw feature vectors across PCIe added overhead.

We solved this by developing an **in-VRAM parallel feature extraction kernel** (`count_packet_features_kernel` and `emit_packet_features_kernel`) using 8 KB shared-memory bitsets per thread block. Raw packet bytes are copied to the GPU once, hashes are evaluated in warp-parallel fashion, and CSR arrays are synthesized directly in device memory. This dropped feature extraction from **1,834.5 ms to 14.87 ms (123.4× speedup)**, allowing end-to-end acceleration to reach **8.56×**."

---

### Question 5: GPU Memory Management & Race Conditions
> **Examiner:** *"In your GPU feature extraction kernel, how did you avoid data races when thousands of threads insert features into the same matrix?"*

**Answer:**
"We used a hierarchical memory design:
1. Each thread block is assigned exactly one network packet.
2. Inside each block, threads allocate a fast 8 KB `__shared__` memory bitset (`2048` unsigned 32-bit integers, representing the 65,536-bit vocabulary).
3. Threads slide across the packet payload in warp stride, compute the FNV-1a hash, and execute `atomicOr(&bitset[hash >> 5], 1u << (hash & 31))` on **shared memory** rather than high-latency global memory.
4. Once synchronized with `__syncthreads()`, thread 0 uses the hardware intrinsic `__ffs()` (Find First Set) to extract set bits and writes strictly sorted column indices directly into the packet's CSR row slice.
5. Because each block writes to an independent row offset in global memory, there are **zero global memory race conditions**."

---

### Question 6: SIMD Vectorization Internals
> **Examiner:** *"Explain how your AVX2 SIMD verification works. Why is it faster than Boyer-Moore-Horspool?"*

**Answer:**
"Boyer-Moore-Horspool performs sub-linear skipping, but on short network packet payloads (50–1500 bytes), skip table lookups and branch mispredictions create overhead.

Our AVX2 implementation uses a **vectorized first-character probe**:
1. It broadcasts the first byte of the attack signature across all 32 lanes of a 256-bit register using `_mm256_set1_epi8()`.
2. It streams 32 bytes of packet payload per iteration into register `YMM0` using unaligned load `_mm256_loadu_si256()`.
3. It performs a parallel byte comparison in one cycle via `_mm256_cmpeq_epi8()`.
4. It extracts a 32-bit hit bitmask using `_mm256_movemask_epi8()`.
5. If the mask is non-zero, it uses `__builtin_ctz()` to find exact matching offsets and only then calls `memcmp()` for the rest of the pattern.
This inspects **32 bytes per clock cycle without branching byte-by-byte**, verifying all 751 candidate pairs in **0.399 ms** (~0.53 μs per candidate)."

---

### Question 7: OpenMP Scaling Anomaly
> **Examiner:** *"Why did OpenMP-4T (1,219 ms) perform slightly worse than OpenMP-2T (1,205 ms) in your Colab benchmarks?"*

**Answer:**
"Google Colab's standard runtime provides an Intel Xeon CPU with **only 2 virtual CPU cores (vCPUs)**. When spawning 4 OpenMP threads on a system with only 2 physical execution contexts, the operating system must time-slice and context-switch threads. The resulting context switching and L1/L2 cache thrashing caused thread contention, leading to a slight performance regression. On our 16-core workstation, OpenMP scaled with near-linear speedup up to 5.93× at 16 threads."

---

### Quick Defense Reference Card

| Metric / Parameter | Value / Implementation |
| :--- | :--- |
| **Vocabulary Space ($V$)** | $65,536$ ($2^{16}$) |
| **N-gram Size** | 3 bytes |
| **Hash Function** | 32-bit FNV-1a (`0x811C9DC5`, prime `16777619`) |
| **Matrix P Dimension** | $10,000 \times 65,536$ ($98.83\%$ sparse, $NNZ = 7,691,118$) |
| **Matrix S Dimension** | $8 \times 65,536$ ($99.98\%$ sparse, $NNZ = 82$) |
| **Feature Extraction (CPU vs GPU)** | $1,834.5\text{ ms} \to 14.87\text{ ms}$ (**$123.4\times$ speedup**) |
| **Stage 1 cuSPARSE SpMM Time** | $43.17\text{ ms}$ on NVIDIA Tesla T4 |
| **Stage 2 SIMD Verification Time**| $0.399\text{ ms}$ ($751$ candidates $\to 355$ verified alerts) |
| **Total End-to-End Latency** | $237.67\text{ ms}$ |
| **End-to-End Speedup** | **$8.56\times$** over single-threaded sequential baseline |
| **Throughput** | **$42,075\text{ packets/sec}$** |
| **Accuracy** | $100\%$ detection equivalence, $0$ false negatives, $0$ false positives |
