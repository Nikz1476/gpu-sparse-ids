# GPU-Accelerated Sparse Linear Algebra Network Intrusion Detection System

A research-oriented, HPC-focused IDS that uses **sparse matrix multiplication** as its central computational mechanism to accelerate signature-based network intrusion detection.

## Mathematical Formulation

```
P  ∈ ℝ^{N×V}   — packet feature matrix   (N packets, V vocab)
S  ∈ ℝ^{K×V}   — signature feature matrix (K signatures, V vocab)
C  =  P × Sᵀ   ∈ ℝ^{N×K}  — score matrix
```

- Each row of **P** contains binary n-gram feature indicators for one packet payload.
- Each row of **S** contains binary n-gram feature indicators for one signature pattern.
- **C[i][j]** = number of features shared between packet *i* and signature *j*.
- A packet–signature pair becomes a **candidate** when `C[i][j] ≥ num_features(j)`, meaning every signature n-gram feature appears in the packet.
- Short signatures (pattern length < n-gram size) **bypass** the sparse filter and go directly to exact verification (safe by design — no false negatives).
- All candidates undergo **exact byte-level substring verification**. Only confirmed matches become alerts.

## Project Structure

```
gpu_sparse_ids/
├── include/
│   ├── common/        types.h          — Packet, Signature, CSRMatrix, Alert
│   ├── ingestion/     pcap_reader.h
│   ├── signatures/    signature_db.h
│   ├── features/      ngram_features.h
│   ├── sparse/        csr_matrix.h
│   ├── compute/       cpu_engine.h, openmp_engine.h, cuda_engine.h
│   └── output/        alert_manager.h
├── src/
│   ├── ingestion/     pcap_reader.cpp
│   ├── signatures/    signature_db.cpp
│   ├── features/      ngram_features.cpp
│   ├── sparse/        csr_matrix.cpp
│   ├── compute/       cpu_engine.cpp, openmp_engine.cpp, cuda_engine_stub.cpp
│   ├── cuda/          cuda_engine.cu   (CUDA/cuSPARSE, requires nvcc)
│   ├── output/        alert_manager.cpp
│   ├── main.cpp
│   ├── benchmark.cpp
│   └── generate_test_data.cpp
├── tests/             test_ids.cpp     — 23-test correctness suite (unit, property, hardened PCAP, and E2E)
├── data/
│   ├── pcap/          test_traffic.pcap, benchmark_traffic.pcap
│   └── signatures/    test_signatures.txt
├── benchmarks/        benchmark_results.csv, *.png
├── scripts/           plot_benchmarks.py, colab_cuda_run.py
├── docs/              architecture.md, technical_report.md
└── CMakeLists.txt
```

## Requirements

| Component | Minimum | Notes |
|-----------|---------|-------|
| g++ | 11+ | C++17 required |
| CMake | 3.16+ | |
| libpcap-dev | 1.8+ | `apt install libpcap-dev` |
| OpenMP | 4.5+ | Included with GCC |
| Python 3 | 3.8+ | For plotting |
| matplotlib | any | Optional, `pip install matplotlib` |
| nvcc / CUDA | 11.0+ | GPU only, optional |

## Installation

```bash
# Debian/Ubuntu/Kali WSL
sudo apt update
sudo apt install -y build-essential cmake libpcap-dev git python3-pip
pip3 install matplotlib numpy
```

## Build

### CPU Build (Sequential + OpenMP)

```bash
git clone <repo_url> gpu_sparse_ids
cd gpu_sparse_ids
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### CUDA Build (requires NVIDIA GPU + CUDA toolkit)

```bash
mkdir build_cuda && cd build_cuda
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON \
         -DCMAKE_CUDA_ARCHITECTURES=75   # T4=75, V100=70, A100=80
make -j$(nproc)
```

## Generate Test Data

```bash
# From repo root (after build)
mkdir -p data/pcap data/signatures benchmarks
./build/generate_test_data data
```

This creates:
- `data/pcap/test_traffic.pcap` — 15 packets with known ground-truth matches
- `data/pcap/benchmark_traffic.pcap` — 10,000 packets with ~5% injected signatures
- `data/signatures/test_signatures.txt` — 8 attack signatures
- `data/expected_matches.txt` — ground-truth reference

## Usage

### Sequential CPU Mode

```bash
./build/sparse_ids data/pcap/test_traffic.pcap data/signatures/test_signatures.txt seq
```

### OpenMP Parallel Mode

```bash
# Auto thread count
./build/sparse_ids data/pcap/test_traffic.pcap data/signatures/test_signatures.txt omp

# 4 threads
./build/sparse_ids data/pcap/test_traffic.pcap data/signatures/test_signatures.txt omp 4
```

### CUDA GPU Mode

```bash
# Requires CUDA build
./build_cuda/sparse_ids data/pcap/test_traffic.pcap data/signatures/test_signatures.txt cuda
```

**Note:** If CUDA is unavailable, the system automatically falls back to sequential CPU execution and reports the fallback.

## Run Tests

```bash
./build/test_ids
```

Expected output: **22 passed, 0 failed**

The test suite verifies:
- N-gram feature extraction (empty, short, deterministic, unique+sorted)
- CSR structure invariants (monotonicity, row_ptr consistency)
- SpMM correctness (known products, sparse vs dense reference on random matrices)
- Candidate generation (threshold filtering, short-signature bypass)
- Exact verification (match, non-match, protocol filter, port filter)
- No-false-negatives property for all pattern lengths
- Short signature end-to-end detection
- Hash collision rate and effect
- End-to-end sequential ground-truth validation
- OpenMP produces identical alerts at 1, 2, and 4 threads

## Benchmarking

```bash
./build/benchmark_ids \
    data/pcap/benchmark_traffic.pcap \
    data/signatures/test_signatures.txt \
    benchmarks/benchmark_results.csv
```

Then visualise:
```bash
python3 scripts/plot_benchmarks.py benchmarks/benchmark_results.csv
```

## Actual Benchmark Results (WSL2, Intel Core i7, 16 logical cores)

| Mode | Threads | Total (ms) | Pkts/sec | Speedup | Candidates | Alerts |
|------|---------|-----------|---------|---------|-----------|--------|
| Sequential | 1 | 916.7 | 10,908 | 1.00× | 751 | 355 |
| OpenMP | 1 | 978.3 | 10,222 | 0.94× | 751 | 355 |
| OpenMP | 2 | 550.2 | 18,176 | 1.67× | 751 | 355 |
| OpenMP | 4 | 327.9 | 30,494 | 2.80× | 751 | 355 |
| OpenMP | 16 | 179.1 | 55,837 | 5.12× | 751 | 355 |

**Matrix dimensions** (10,000 packets, 8 signatures, vocab=65536):
- P: 10,000 × 65,536 — nnz = 7,691,118 — sparsity = 98.83%
- S: 8 × 65,536 — nnz = 82 — sparsity = 99.98%
- C: 10,000 × 8 — nnz = 9,254

> CUDA benchmark: **not yet validated** — no local NVIDIA GPU. See `scripts/colab_cuda_run.py` for reproducible GPU execution on Google Colab.

## Running CUDA on Google Colab

```python
# In a Colab cell (Runtime: GPU → T4):
!pip install -q matplotlib
!git clone https://github.com/YOUR_USERNAME/gpu_sparse_ids.git
%cd gpu_sparse_ids
!python3 scripts/colab_cuda_run.py
```

Or follow `scripts/colab_cuda_run.py` manually.

## Signature File Format

```
# ID|NAME|PROTOCOL|PORT|HEX_PATTERN|DESCRIPTION
# PROTOCOL: 0=any, 6=TCP, 17=UDP   PORT: 0=any
1001|SQL_INJECTION_OR|6|0|2720f52520315df31d2d2d|SQL injection OR 1=1
```

## Known Limitations

| Limitation | Detail |
|-----------|--------|
| IPv4 only | IPv6 packets are silently skipped |
| No TCP reassembly | Multi-packet payloads are not reassembled |
| No VLAN/tunnelling | 802.1Q, GRE, VXLAN not supported |
| Hash collisions | FNV-1a with vocab=65536 gives ~8% collision rate on 3-grams; collisions increase candidates, never cause false negatives |
| Short signatures | Patterns shorter than n-gram size bypass sparse filter (all-pairs scan); acceptable for small signature sets |
| PCAP only | Offline processing; no live capture |
| No fragmented IP | Fragmented IP packets skipped |

## References

- Gnort (GPU Snort): https://doi.org/10.1007/978-3-540-87403-4_7
- Kargus: https://doi.org/10.1145/2382196.2382232
- GPU Aho-Corasick: https://arxiv.org/abs/1704.02272
- CIC-IDS2017: https://www.unb.ca/cic/datasets/ids-2017.html
- NVIDIA cuSPARSE: https://docs.nvidia.com/cuda/cusparse/

