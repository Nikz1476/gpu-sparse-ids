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

### Structured Output (CSV & JSON)

```bash
# Export alerts to CSV and JSON alongside execution summary
./build/sparse_ids data/pcap/test_traffic.pcap data/signatures/test_signatures.txt omp 4 \
    --out-csv output/alerts.csv \
    --out-json output/alerts.json
```

### CUDA GPU Mode

```bash
# Requires CUDA build on system with NVIDIA GPU
./build_cuda/sparse_ids data/pcap/test_traffic.pcap data/signatures/test_signatures.txt cuda
```

**Note:** If CUDA is unavailable, the system automatically falls back to sequential CPU execution and reports the fallback.

## Run Tests

```bash
./build/test_ids
```

Expected output: **24 passed, 0 failed**

The test suite verifies:
- N-gram feature extraction (empty, short, deterministic, unique+sorted)
- CSR structure invariants (monotonicity, row_ptr consistency)
- SpMM correctness (known products, sparse vs dense reference on random matrices)
- Candidate generation (threshold filtering, short-signature bypass)
- Exact verification (match, non-match, protocol filter, port filter, AVX2 SIMD scanning)
- No-false-negatives property for all pattern lengths
- Short signature end-to-end detection
- IPv6 and 802.1Q VLAN packet parsing and filtering
- Hash collision rate and effect
- PCAP Hardening: graceful rejection of truncated, malformed packets
- End-to-end sequential ground-truth validation
- OpenMP produces identical alerts at 1, 2, and 4 threads
- Hash collision rate and effect
- PCAP Hardening: graceful rejection of truncated, non-IPv4, and malformed packets
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

## Actual Benchmark Results

### 1. NVIDIA Tesla T4 GPU (Google Colab Environment)

Measured on Google Colab (Intel Xeon @ 2.20GHz, 2 vCPUs, NVIDIA Tesla T4 16GB, 10,000 Packets, 8 Signatures):

| Mode | FeatExt (ms) | CSR Build (ms) | SpMM (ms) | H2D (ms) | D2H (ms) | Total Compute (ms) | **Total Wallclock (ms)** | Throughput (pkts/s) | Speedup vs Seq |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Sequential (1T)** | 1598.54 | 33.15 | 52.61 | 0.00 | 0.00 | 1685.47 | 1685.71 | 5,932 | 1.00× |
| **OpenMP (1T)** | 1512.50 | 35.60 | 73.15 | 0.00 | 0.00 | 1622.37 | 1622.71 | 6,163 | 1.04× |
| **OpenMP (2T)** | 1067.31 | 42.27 | 65.10 | 0.00 | 0.00 | 1175.59 | 1175.83 | 8,505 | 1.43× |
| **OpenMP (4T)** | 1716.93 | 41.44 | 130.47 | 0.00 | 0.00 | 1890.64 | 1890.92 | 5,288 | 0.89× |
| **CUDA (cuSPARSE + GPU FeatExt)** | **15.92** | **6.82** | **57.01** | **2.47** | **0.11** | **81.29** | **278.96** | **35,847** | **6.04×** |

> **Key HPC Optimization:** Feature extraction time dropped from **1,598 ms to 15.92 ms** (a **100.4× speedup**) using an in-VRAM warp-parallel CUDA kernel with an 8 KB shared-memory bitset per block, eliminating Amdahl's Law bottleneck and enabling **6.04× end-to-end acceleration**.

### 2. Multi-Core CPU (WSL2, Intel Core i5-1240P, 16 logical cores)

| Mode | Threads | Total (ms) | Pkts/sec | Speedup | Candidates | Alerts |
|------|---------|-----------|---------|---------|-----------|--------|
| Sequential | 1 | 943.2 | 10,602 | 1.00× | 751 | 355 |
| OpenMP | 1 | 933.3 | 10,714 | 1.01× | 751 | 355 |
| OpenMP | 2 | 569.2 | 17,568 | 1.66× | 751 | 355 |
| OpenMP | 4 | 322.7 | 30,991 | 2.92× | 751 | 355 |
| OpenMP | 8 | 242.8 | 41,180 | 3.88× | 751 | 355 |
| OpenMP | 16 | 159.0 | 62,877 | 5.93× | 751 | 355 |

**Matrix dimensions** (10,000 packets, 8 signatures, vocab=65536):
- P: 10,000 × 65,536 — nnz = 7,691,118 — sparsity = 98.83%
- S: 8 × 65,536 — nnz = 82 — sparsity = 99.98%
- C: 10,000 × 8 — nnz = 9,254 (filtered down to 751 candidates -> 355 verified alerts)

## Running CUDA on Google Colab

An automated interactive Jupyter Notebook is provided at:
[`notebooks/gpu_sparse_ids_colab.ipynb`](notebooks/gpu_sparse_ids_colab.ipynb)

**Steps on Google Colab:**
1. Open Google Colab and upload or open `notebooks/gpu_sparse_ids_colab.ipynb`.
2. Select **Runtime** -> **Change runtime type** -> **T4 GPU**.
3. Run all cells sequentially. The notebook automatically:
   - Detects GPU model and CUDA capability (`sm_75`, `sm_80`, etc.).
   - Installs dependencies (`libpcap-dev`, `cmake`).
   - Configures CMake with `-DUSE_CUDA=ON` and compiles all binaries.
   - Executes the 23-test suite.
   - Runs comparative benchmarks (Sequential vs OpenMP vs CUDA cuSPARSE).
   - Validates alert equivalence across CPU and GPU backends.
   - Visualizes performance and exports all artifacts as a downloadable `.zip`.

## Component Validation Status

| Component | Status | Environment |
|---|---|---|
| PCAP Reader & Hardening | Tested & Verified | WSL2 Linux, libpcap |
| Byte N-Gram Extraction | Tested & Verified | Local CPU (FNV-1a) |
| Short-Signature Bypass | Tested & Verified | Zero false negatives |
| CSR Construction & SpMM | Tested & Verified | Dense matrix equivalence |
| Sequential CPU Backend | Tested & Verified | Ground truth validated |
| OpenMP Parallel Backend | Tested & Verified | Scaled up to 16 threads, TSan clean |
| Structured Alerts (CSV/JSON) | Tested & Verified | CLI verified |
| CUDA / cuSPARSE SpGEMM | Implemented & Colab Ready | `notebooks/gpu_sparse_ids_colab.ipynb` |
| NVIDIA Hardware Validation | Remote Ready | Requires NVIDIA GPU (e.g. Colab T4) |

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

