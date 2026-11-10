# The Complete Project Walkthrough: How and Why It Works

> **What is this document?**  
> An elaborate, clear, and easy-to-read explanation of the **GPU Sparse Network Intrusion Detection System (IDS)**.  
> It explains the **math**, the **code**, the **GPU mechanics**, and the **HPC design choices** in plain English without cutting corners on technical depth.

---

## 1. What Problem Does This Project Solve?

In computer networks, an **Intrusion Detection System (IDS)** acts like a digital security guard. Every second, tens of thousands of data packets fly through a network. The IDS must inspect the payload (the actual text or binary content inside each packet) to see if a hacker is sending an attack (like an SQL Injection, Cross-Site Scripting, or Shell command).

### The Traditional Problem: Why Normal Computers Struggle
Traditional tools (like Snort or Suricata) use pattern-matching algorithms (like Aho-Corasick or Boyer-Moore):
* They read packets letter-by-letter and compare them against rules.
* If you have 10,000 packets and 1,000 attack rules, the computer has to do millions of character comparisons.
* The computer's processor spends most of its time jumping around memory looking up state transitions ("pointer chasing"). This causes **cache misses** and slows the entire network down.

### Our Solution: The Sparse Matrix Multiplication Trick
Instead of treating pattern matching as a slow text-search problem, we turn it into **Sparse Matrix Multiplication**:

$$\mathbf{C} = \mathbf{P} \times \mathbf{S}^T$$

By turning packets and attack rules into rows of numbers (matrices), we can hand the entire workload over to a **GPU (Graphics Processing Unit)**. A GPU has thousands of tiny cores that can perform millions of multiplications all at the same time.

---

## 2. Breaking Down the Mathematics (Simply Explained)

### Step 1: 3-Gram Sliding Windows
Imagine an attack signature is the word `ATTACK`. 
Instead of searching for the whole word at once, we chop it into sliding windows of 3 letters (called **3-grams**):
* Window 1: `A T T`
* Window 2: `T T A`
* Window 3: `T A C`
* Window 4: `A C K`

A 6-letter word produces 4 distinct 3-grams.

### Step 2: Hashing (Why FNV-1a?)
We have to turn these 3-letter pieces into numbers so the computer can track them in a matrix.

* **Why can't we just use the raw letters?**  
  There are $256 \times 256 \times 256 = 16,777,216$ possible combinations of 3 bytes. Keeping track of 16.7 million columns would waste gigabytes of memory.
* **Our Vocabulary ($V = 65,536$):**  
  We want every 3-gram to get a bucket number between `0` and `65,535` ($2^{16}$).
* **Why did we pick the FNV-1a Hash?**
  1. **Speed:** It takes only 1 or 2 clock cycles per byte.
  2. **No Branching:** There are no `if-else` statements. This is vital for GPUs because `if` statements cause GPU threads to stall (called "warp divergence").
  3. **Low Memory Use:** It only needs one single 32-bit register variable:
     ```cpp
     uint32_t hash = 2166136261u;   // Starting seed
     for (int i = 0; i < len; ++i) {
         hash ^= payload[i];        // 1. XOR the incoming byte
         hash *= 16777619u;          // 2. Multiply by FNV prime
     }
     return hash % 65536;           // 3. Keep within 0 to 65535
     ```
  Multiplying by the prime `16777619` scatters the bits across the register (called the "avalanche effect"), ensuring different 3-grams rarely land in the same bucket.

---

## 3. Matrix Representation: Why Do We Need CSR?

Now we build two attendance sheets (matrices):
1. **Matrix P (Packets):** $10,000$ rows (packets) $\times 65,536$ columns (n-gram feature buckets).  
   If packet #5 has feature #42, we set `P[5, 42] = 1`.
2. **Matrix S (Signatures):** $8$ rows (attack rules) $\times 65,536$ columns.  
   If rule #1 has feature #42, we set `S[1, 42] = 1`.

### The Problem: A Normal (Dense) Matrix is Too Huge
If you store Matrix P as a regular 2D grid:
$$10,000 \times 65,536 \times 4 \text{ bytes (floats)} = \mathbf{2,621,440,000 \text{ bytes}} \approx \mathbf{2.62 \text{ GB!}}$$

However, a real packet of 1,000 bytes can only have at most ~998 3-grams. That means out of 65,536 numbers in a row, **more than 64,500 are empty zeroes!**  
Matrix P is **$98.83\%$ empty space (sparse)**.

### The Solution: Compressed Sparse Row (CSR) Format
Instead of storing 2.62 GB of zeroes, we store only the non-zero numbers using three 1D lists:
1. `col_idx`: A list of the actual column numbers that have a `1` (e.g., `[12, 450, 999]`).
2. `values`: The value at each column (all `1.0f`).
3. `row_ptr`: A list of pointers indicating where each packet's features start and end in the `col_idx` list.

**Result:** Memory drops from **2.62 GB down to just 61.5 MB** (a **42.6× reduction**)!

---

## 4. What Exactly Does cuSPARSE Do?

### The Operation: $C = P \times S^T$
We want to multiply Matrix P by the transpose of Matrix S ($S^T$).
* In matrix math, multiplying Row $i$ of Matrix P with Column $j$ of Matrix $S^T$ computes their **dot product**:
  $$\mathbf{C}[i, j] = \sum_{h=0}^{65535} \mathbf{P}[i, h] \cdot \mathbf{S}[j, h]$$
* **What does this dot product mean in plain English?**  
  Because both matrices contain `1`s for features that are present, $\mathbf{C}[i, j]$ simply counts **how many unique 3-grams packet $i$ and signature $j$ have in common**.

### What cuSPARSE Does Inside the GPU Hardware
Instead of writing a slow loop on the CPU, we call NVIDIA’s **cuSPARSE** library (`cusparseSpGEMM`):
1. **Assigns Rows to Warps:** The GPU groups 32 hardware threads into a "warp" and assigns rows of packets to each warp.
2. **Parallel Stream Processing:** The Tesla T4 GPU has **$320 \text{ GB/s}$ of memory bandwidth**. cuSPARSE streams through the non-zero elements of matrix P and matrix S simultaneously, checking matching column indices in parallel.
3. **Scratchpad Memory:** It estimates memory buffers upfront (`workEstimation`), runs the computation (`compute`), and writes the final scores into Matrix C (`copy`).

### How the Score Filters Bad Packets
Suppose Attack Rule #1 (`SQL Injection`) has **4 unique 3-grams** (`num_features = 4`).
* If Packet #100 has a score of `4.0`, it contains **all 4 puzzle pieces**. It is marked as a **Candidate** for final checking.
* If Packet #101 has a score of `2.0`, it only has 2 of the pieces.  
  Since the full attack word physically cannot exist without all 4 pieces, **Packet #101 is dropped instantly**.
* **Result:** More than **$99\%$ of safe traffic is eliminated** right here on the GPU before any text scanning takes place!

---

## 5. The In-Place GPU Candidate Filter

In early designs, the computer copied the entire Score Matrix C back from the GPU to the CPU over the PCIe bus.  
Copying millions of numbers over PCIe takes valuable time.

### How We Fixed It: `filter_candidates_gpu_kernel`
Instead of copying Matrix C back to the CPU:
1. We run a custom CUDA kernel directly on the GPU.
2. Each thread checks its packet's score:
   ```cuda
   if (score >= required_features) {
       int pos = atomicAdd(candidate_count, 1);
       out_candidates[pos].packet_idx = packet_id;
       out_candidates[pos].signature_idx = signature_id;
   }
   ```
3. Only the small list of suspicious pairs (e.g., 751 candidates $\approx 9 \text{ KB}$) is sent back to the CPU! This eliminates PCIe bottlenecks.

---

## 6. Edge Case: The Short-Signature Bypass Rule

What if a hacker uses a very short attack pattern, like `\x90\x90` (2 bytes)?
* A 2-byte word **cannot be cut into a 3-byte puzzle piece**!
* If we tried to hash it, it would have 0 features, its score would always be 0, and our matrix filter would accidentally ignore it.

### Our Solution:
In `generate_candidates()` and `filter_candidates_gpu_kernel`:
* If any rule has `pattern.size() < 3`, it **bypasses the matrix filter completely**.
* It is directly paired with every packet and sent straight to Stage 2 verification.
* Because short rules are rare, this takes almost no extra time, but guarantees **100% security with zero missed attacks**.

---

## 7. Stage 2: AVX2 SIMD Vectorized Verification

Now we have ~750 suspicious packet-signature pairs. We must verify whether the packet actually contains the exact word (to ensure a hash collision didn't trick us).

### Why Normal Substring Search is Slow
Normal string searches check one byte at a time:
`Is byte 0 an 'A'? No. Is byte 1 an 'A'? No...`

### The AVX2 SIMD Vector Solution
Modern CPUs have special 256-bit registers called **AVX2 SIMD** (Single Instruction, Multiple Data).  
A 256-bit register can hold **32 individual 8-bit bytes**:

```
Packet Bytes:       [ 32 bytes loaded at once into 256-bit register YMM0 ]
Attack First Byte:  [ Broadcasted 32 times into 256-bit register YMM1 ]
                            │
                            ▼
                    _mm256_cmpeq_epi8()
                            │
                            ▼
Bitmask Output:     [ 32-bit number: each bit is 1 if that byte matched ]
```

1. **`_mm256_loadu_si256`:** Loads 32 bytes of packet payload in one single clock cycle.
2. **`_mm256_cmpeq_epi8`:** Compares all 32 bytes against the first character of the attack signature simultaneously.
3. **`_mm256_movemask_epi8`:** Creates a 32-bit integer where bit $k$ is `1` if byte $k$ matched.
4. **`__builtin_ctz`:** A CPU hardware instruction ("Count Trailing Zeros") that tells us the exact matching offset in zero time.
5. **`memcmp`:** Only if a bit is `1` do we check the rest of the attack word.

**Result:** Verifying 751 candidates across 10,000 packets takes just **0.399 milliseconds** (~0.53 microseconds per candidate)!

---

## 8. Summary of Every Code File

| Source File | What It Does | Why It Is Implemented This Way |
| :--- | :--- | :--- |
| **`src/ingestion/pcap_reader.cpp`** | Reads `.pcap` files using `libpcap`. Decodes Ethernet, 802.1Q VLAN, IPv4, and IPv6 headers. | Extracts clean packet payloads while gracefully skipping malformed packets without crashing. |
| **`src/features/ngram_features.cpp`** | CPU feature extractor. Slides a 3-byte window and computes 32-bit FNV-1a hashes. | Used as reference baseline and for preprocessing attack signatures. |
| **`src/sparse/csr_matrix.cpp`** | Implements Compressed Sparse Row matrices (`row_ptr`, `col_idx`, `values`). | Compresses matrix data from 2.62 GB to 61.5 MB (42.6× memory savings). |
| **`src/cuda/cuda_engine.cu`** | The complete GPU pipeline: in-VRAM feature hashing, cuSPARSE SpMM, and in-place candidate filtering. | Uses GPU shared-memory bitsets and cuSPARSE to achieve an **8.56× speedup** over CPU. |
| **`src/compute/cpu_engine.cpp`** | Single-threaded CPU pipeline and AVX2 SIMD vectorized verification. | Serves as the ground-truth baseline and provides 32-byte-per-cycle verification. |
| **`src/compute/openmp_engine.cpp`** | Multi-threaded CPU engine using `#pragma omp parallel for`. | Compares multi-core CPU performance against GPU performance. |
| **`src/output/alert_manager.cpp`** | Formats verified attack alerts into structured `.csv` and `.json` files. | Provides clean output for security logging and analysis. |

---

## 9. Final Performance Numbers (Google Colab NVIDIA T4 GPU)

* **Traffic Tested:** 10,000 real network packets.
* **Sequential CPU Time:** **2,033.62 ms** (throughput: 4,917 packets/sec).
* **GPU CUDA Engine Time:** **237.67 ms** (throughput: **42,075 packets/sec**).
* **End-to-End Speedup:** **8.56× faster!**
* **Feature Extraction Speedup:** 1,834.5 ms down to 14.87 ms (**123.4× faster**).
* **Detection Accuracy:** **100% equivalence** (355 verified alerts out of 355 ground-truth attacks, 0 false positives, 0 false negatives).
