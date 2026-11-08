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
- 22-test suite: unit tests + dense-reference SpMM validation + end-to-end ground truth

## Slide 4 — Results (40 sec)
- 10,000 packets, 8 signatures, vocab 65,536
- P: 10,000 × 65,536 — 98.83% sparse · S: 8 × 65,536 — 99.98% sparse
- Sequential: 917 ms · OpenMP-4T: 328 ms (2.8×) · OpenMP-16T: 179 ms (5.1×)
- Dominant bottleneck: **feature extraction** (90% of time) — not SpMM

## Slide 5 — Roofline & GPU Outlook (20 sec)
- SpMM arithmetic intensity: ~1.01 ops/byte → memory-bound
- GPU SpMM alone won't achieve large end-to-end speedup
- **Real GPU speedup requires** GPU-side n-gram hashing + cuSPARSE SpMM together
- CUDA code ready; validation pending free GPU (Colab script provided)

## Slide 6 — Conclusion (10 sec)
- Sparse n-gram IDS: correct, parallelisable, research-validated
- CPU OpenMP gives real 5× speedup; GPU path architecturally complete
- Next: GPU feature extraction kernel + CIC-IDS2017 evaluation

