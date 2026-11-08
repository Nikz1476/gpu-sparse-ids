# Three-Minute Presentation Outline — GPU Sparse IDS

## Slide 1 — Problem (20 sec)
- Network IDS must inspect every packet against hundreds of signatures in real time
- Traditional: regex automata (Aho-Corasick) — complex, monolithic
- **Our approach:** reformulate as sparse matrix multiplication

## Slide 2 — Mathematical Formulation (40 sec)
- Packet feature matrix P (N×V): binary n-gram presence
- Signature feature matrix S (K×V): same
- Score matrix **C = P × Sᵀ**: C[i][j] = shared features between packet i and signature j
- Candidate: `C[i][j] ≥ num_features(j)` → then exact byte verification
- Key property: **no false negatives** for exact contiguous matches

## Slide 3 — Implementation (30 sec)
- CSR format for all sparse matrices
- Three execution paths: Sequential CPU · OpenMP CPU · CUDA/cuSPARSE GPU
- Short signature bypass: patterns < n-gram size skip sparse filter, go direct to verify
- 23-test suite: unit tests + hardened PCAP rejection + dense-reference SpMM validation + end-to-end ground truth
- Structured reporting: Terminal, CSV, and full JSON execution summaries

## Slide 4 — Results (40 sec)
- 10,000 packets, 8 signatures, vocab 65,536
- P: 10,000 × 65,536 — 98.83% sparse · S: 8 × 65,536 — 99.98% sparse
- Sequential: 943 ms (10,602 pkts/s) · OpenMP-4T: 323 ms (2.92×) · OpenMP-16T: 159 ms (5.93×, 62,877 pkts/s)
- Dominant bottleneck: **feature extraction** (70-90% of time) — not SpMM
- ThreadSanitizer verified: zero data races across multi-threaded extraction and SpMM

## Slide 5 — Roofline & GPU Outlook (20 sec)
- SpMM arithmetic intensity: ~1.0127 ops/byte → memory-bandwidth bound
- Best measured CPU SpMM throughput: 11.36 GOps/s (16 threads)
- GPU cuSPARSE SpGEMM backend complete with RAII safety and sorted CSR guarantees
- Interactive Colab notebook (`notebooks/gpu_sparse_ids_colab.ipynb`) for 1-click cloud GPU execution
- Real GPU speedup requires GPU-side n-gram hashing + cuSPARSE SpMM together

## Slide 6 — Conclusion (10 sec)
- Sparse n-gram IDS: correct, parallelisable, research-validated
- CPU OpenMP gives real 5× speedup; GPU path architecturally complete
- Next: GPU feature extraction kernel + CIC-IDS2017 evaluation

