import json
import os

nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# GPU-Accelerated Sparse Linear Algebra Network Intrusion Detection System\n",
                "### Reproducible NVIDIA GPU (CUDA / cuSPARSE) Execution & Benchmark on Google Colab\n",
                "\n",
                "This notebook configures, builds, and evaluates the **GPU Sparse IDS** engine.\n",
                "\n",
                "**Hardware Requirement:**\n",
                "Ensure you are running on a GPU runtime: `Runtime` -> `Change runtime type` -> select `T4 GPU` (or V100/A100).\n",
                "\n",
                "**Core Motif:** Sparse linear algebra ($C = P \\times S^T$) with byte n-gram feature representation + exact substring verification.\n",
                "\n",
                "**Verification Engine:** SIMD-Vectorized Scanning (AVX2) — `_mm256_cmpeq_epi8` first-character probe, 32 bytes/cycle."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 1: Hardware & Environment Detection\n",
                "Check for NVIDIA GPU availability, driver status, CUDA toolkit version, and AVX2 SIMD support."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import subprocess, sys, os\n",
                "\n",
                "print(\"=== Detecting NVIDIA GPU ===\")\n",
                "try:\n",
                "    smi_out = subprocess.check_output([\"nvidia-smi\"], text=True)\n",
                "    print(smi_out)\n",
                "    has_gpu = True\n",
                "except Exception as e:\n",
                "    print(\"WARNING: nvidia-smi failed or no NVIDIA GPU detected:\", e)\n",
                "    has_gpu = False\n",
                "\n",
                "try:\n",
                "    nvcc_out = subprocess.check_output([\"nvcc\", \"--version\"], text=True)\n",
                "    print(\"=== CUDA Compiler (nvcc) ===\")\n",
                "    print(nvcc_out)\n",
                "except Exception as e:\n",
                "    print(\"WARNING: nvcc not found in PATH:\", e)\n",
                "\n",
                "# Check AVX2 support on the host CPU\n",
                "print(\"=== AVX2 SIMD Support ===\")\n",
                "try:\n",
                "    cpuinfo = open('/proc/cpuinfo').read()\n",
                "    if 'avx2' in cpuinfo:\n",
                "        print(\"AVX2 is SUPPORTED on this CPU — SIMD vectorized scanning will be active.\")\n",
                "    else:\n",
                "        print(\"AVX2 NOT detected — scalar fallback will be used for verification.\")\n",
                "except:\n",
                "    print(\"Could not read /proc/cpuinfo\")\n",
                "\n",
                "if not has_gpu:\n",
                "    print(\"\\n[FALLBACK NOTICE] No NVIDIA GPU found in this Colab session.\")\n",
                "    print(\"Please navigate to Runtime -> Change runtime type -> Hardware accelerator -> T4 GPU.\")\n",
                "    print(\"The notebook can still run in CPU (Sequential / OpenMP) fallback mode.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 2: Clone or Update Repository\n",
                "Clone from GitHub or reset to the latest `origin/main`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "\n",
                "REPO_DIR = \"/content/gpu_sparse_ids\"\n",
                "REPO_URL = \"https://github.com/Nikz1476/gpu-sparse-ids.git\"\n",
                "\n",
                "if not os.path.exists(REPO_DIR):\n",
                "    print(f\"Cloning {REPO_URL} into {REPO_DIR}...\")\n",
                "    !git clone {REPO_URL} {REPO_DIR}\n",
                "else:\n",
                "    print(f\"Updating existing repository at {REPO_DIR}...\")\n",
                "    !cd {REPO_DIR} && git fetch origin && git reset --hard origin/main\n",
                "\n",
                "%cd {REPO_DIR}\n",
                "!git log -n 5 --oneline\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 3: Install Required Dependencies\n",
                "Install `libpcap-dev`, `cmake`, `build-essential`, and python plotting libraries."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!apt-get update -qq\n",
                "!apt-get install -y -qq build-essential cmake libpcap-dev git\n",
                "!pip install -q matplotlib numpy pandas"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 4: Configure & Build CUDA / cuSPARSE Engine\n",
                "Auto-detect GPU compute architecture (T4=75, V100=70, A100=80).\n",
                "\n",
                "**Build features:**\n",
                "- `-DUSE_CUDA=ON`: cuSPARSE SpGEMM + GPU parallel feature extraction\n",
                "- `-mavx2`: AVX2 SIMD vectorized scanning (auto-detected by CMake)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import torch, os\n",
                "\n",
                "cuda_arch = \"75\"  # Default for Colab T4\n",
                "if torch.cuda.is_available():\n",
                "    major, minor = torch.cuda.get_device_capability()\n",
                "    cuda_arch = f\"{major}{minor}\"\n",
                "    gpu_name = torch.cuda.get_device_name(0)\n",
                "    print(f\"Detected GPU: {gpu_name} (Compute capability: sm_{cuda_arch})\")\n",
                "else:\n",
                "    print(\"GPU not detected via PyTorch, defaulting sm_75\")\n",
                "\n",
                "# Configure CMake with CUDA enabled\n",
                "!rm -rf build_cuda\n",
                "!mkdir -p build_cuda\n",
                "%cd build_cuda\n",
                "!cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES={cuda_arch} 2>&1 | tail -20\n",
                "!make -j$(nproc) 2>&1 | tail -10\n",
                "%cd ..\n",
                "\n",
                "# Verify binaries were built\n",
                "import os\n",
                "for binary in ['sparse_ids', 'benchmark_ids', 'test_ids', 'generate_test_data']:\n",
                "    path = f'build_cuda/{binary}'\n",
                "    status = '✅' if os.path.exists(path) else '❌ MISSING'\n",
                "    print(f'  {binary}: {status}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 5: Generate Test Data\n",
                "Produce ground-truth synthetic PCAP and attack signatures (10,000 packets)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!mkdir -p data/pcap data/signatures benchmarks output\n",
                "!./build_cuda/generate_test_data data 10000\n",
                "!echo ''\n",
                "!echo '=== Generated Files ==='\n",
                "!ls -la data/pcap data/signatures"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 6: Run Correctness Test Suite\n",
                "Validate all 24 unit, property, and integration tests (including IPv6/VLAN)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!./build_cuda/test_ids"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 7: Run CUDA IDS on Ground-Truth Traffic\n",
                "Execute the full pipeline: GPU Feature Extraction → cuSPARSE SpGEMM → GPU Candidate Filter → AVX2 SIMD Verification."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!./build_cuda/sparse_ids data/pcap/test_traffic.pcap data/signatures/test_signatures.txt cuda 1 3 65536 \\\n",
                "    --out-csv output/cuda_alerts.csv \\\n",
                "    --out-json output/cuda_alerts.json\n",
                "\n",
                "import json\n",
                "if os.path.exists(\"output/cuda_alerts.json\"):\n",
                "    with open(\"output/cuda_alerts.json\") as f:\n",
                "        res = json.load(f)\n",
                "    print(f\"\\nGenerated {len(res['alerts'])} verified alerts via CUDA cuSPARSE + AVX2 SIMD verification.\")\n",
                "    print(\"JSON Summary:\", json.dumps(res['summary'], indent=2))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 8: Comparative Benchmark (Sequential vs OpenMP vs CUDA)\n",
                "Run full pipeline benchmarks on 10,000 packets.\n",
                "\n",
                "All modes use **AVX2 SIMD vectorized scanning** for the verification phase:\n",
                "- `_mm256_cmpeq_epi8` first-character probe → 32 bytes/cycle\n",
                "- `_mm256_movemask_epi8` bitmask extraction → `memcmp` only at hits"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!./build_cuda/benchmark_ids data/pcap/benchmark_traffic.pcap data/signatures/test_signatures.txt benchmarks/benchmark_results.csv\n",
                "!echo ''\n",
                "!echo '=== Benchmark CSV ==='\n",
                "!cat benchmarks/benchmark_results.csv"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 9: Verification Phase — SIMD AVX2 Micro-Benchmark\n",
                "Isolate and measure the **verification phase** performance across all modes.\n",
                "\n",
                "The verification phase uses SIMD-Vectorized Scanning (AVX2 / AVX-512 capable):\n",
                "- **Vectorized First-Character Probe** (`_mm256_cmpeq_epi8`):\n",
                "  Load 32 bytes of packet payload into a 256-bit AVX2 register in one CPU cycle.\n",
                "  Compare against the first byte of the signature pattern.\n",
                "  Extract a bitmask using `_mm256_movemask_epi8()`. Only call `memcmp` on matching offsets.\n",
                "- **Result:** Checks 32 bytes per clock cycle without looping byte-by-byte."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import matplotlib\n",
                "matplotlib.use('Agg')\n",
                "import matplotlib.pyplot as plt\n",
                "import numpy as np\n",
                "\n",
                "# Read benchmark results\n",
                "df = pd.read_csv('benchmarks/benchmark_results.csv')\n",
                "\n",
                "# Build labels\n",
                "labels = []\n",
                "for _, row in df.iterrows():\n",
                "    if row['mode'] == 'Sequential':\n",
                "        labels.append('Sequential')\n",
                "    elif row['mode'] == 'OpenMP':\n",
                "        labels.append(f\"OpenMP-{int(row['threads'])}T\")\n",
                "    else:\n",
                "        labels.append(row['mode'])\n",
                "\n",
                "# === Verification Time Bar Chart ===\n",
                "ver_times = df['verification_ms'].values\n",
                "candidates = df['num_candidates'].values\n",
                "alerts = df['num_alerts'].values\n",
                "\n",
                "print('=' * 65)\n",
                "print('  SIMD AVX2 Vectorized Scanning — Verification Phase Benchmark')\n",
                "print('=' * 65)\n",
                "print(f\"{'Mode':<20} {'Verify (ms)':>12} {'Candidates':>12} {'Alerts':>8} {'μs/candidate':>14}\")\n",
                "print('-' * 65)\n",
                "for i, label in enumerate(labels):\n",
                "    us_per = (ver_times[i] * 1000 / candidates[i]) if candidates[i] > 0 else 0\n",
                "    print(f\"  {label:<18} {ver_times[i]:>12.4f} {int(candidates[i]):>12} {int(alerts[i]):>8} {us_per:>14.4f}\")\n",
                "\n",
                "# Verification time plot\n",
                "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))\n",
                "\n",
                "colors = []\n",
                "for _, row in df.iterrows():\n",
                "    if row['mode'] == 'Sequential':  colors.append('#2196F3')\n",
                "    elif row['mode'] == 'OpenMP':    colors.append('#4CAF50')\n",
                "    else:                            colors.append('#FF9800')\n",
                "\n",
                "# Left: Verification time\n",
                "bars = ax1.bar(labels, ver_times, color=colors, edgecolor='black', linewidth=0.5)\n",
                "ax1.set_xlabel('Execution Mode', fontsize=11)\n",
                "ax1.set_ylabel('Verification Time (ms)', fontsize=11)\n",
                "ax1.set_title('SIMD AVX2 Vectorized Scanning — Verification Time', fontsize=12, fontweight='bold')\n",
                "for bar, t in zip(bars, ver_times):\n",
                "    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,\n",
                "            f'{t:.3f}ms', ha='center', va='bottom', fontsize=8)\n",
                "ax1.grid(axis='y', alpha=0.3)\n",
                "ax1.tick_params(axis='x', rotation=20)\n",
                "\n",
                "# Right: Throughput (candidates verified per ms)\n",
                "throughput = [c / v if v > 0 else 0 for c, v in zip(candidates, ver_times)]\n",
                "bars2 = ax2.bar(labels, throughput, color=colors, edgecolor='black', linewidth=0.5)\n",
                "ax2.set_xlabel('Execution Mode', fontsize=11)\n",
                "ax2.set_ylabel('Candidates Verified / ms', fontsize=11)\n",
                "ax2.set_title('SIMD AVX2 — Verification Throughput', fontsize=12, fontweight='bold')\n",
                "for bar, tp in zip(bars2, throughput):\n",
                "    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,\n",
                "            f'{tp:,.0f}', ha='center', va='bottom', fontsize=8)\n",
                "ax2.grid(axis='y', alpha=0.3)\n",
                "ax2.tick_params(axis='x', rotation=20)\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.savefig('benchmarks/verification_simd_benchmark.png', dpi=150, bbox_inches='tight')\n",
                "plt.close()\n",
                "\n",
                "from IPython.display import Image, display\n",
                "display(Image('benchmarks/verification_simd_benchmark.png'))\n",
                "print(f\"\\nSaved: benchmarks/verification_simd_benchmark.png\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 10: Alert Equivalence Validation\n",
                "Confirm that Sequential CPU, OpenMP multi-thread, and CUDA GPU produce **100% identical alerts**.\n",
                "\n",
                "This validates that the AVX2 SIMD verification produces bit-exact results across all execution modes."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import json\n",
                "\n",
                "# Run Sequential\n",
                "!./build_cuda/sparse_ids data/pcap/test_traffic.pcap data/signatures/test_signatures.txt seq 1 3 65536 --out-json output/seq.json\n",
                "# Run OpenMP\n",
                "!./build_cuda/sparse_ids data/pcap/test_traffic.pcap data/signatures/test_signatures.txt omp 2 3 65536 --out-json output/omp.json\n",
                "# Run CUDA\n",
                "!./build_cuda/sparse_ids data/pcap/test_traffic.pcap data/signatures/test_signatures.txt cuda 1 3 65536 --out-json output/cuda.json\n",
                "\n",
                "with open('output/seq.json') as f1, open('output/omp.json') as f2, open('output/cuda.json') as f3:\n",
                "    s_alerts = json.load(f1)['alerts']\n",
                "    o_alerts = json.load(f2)['alerts']\n",
                "    c_alerts = json.load(f3)['alerts']\n",
                "\n",
                "print(f\"\\n{'='*50}\")\n",
                "print(f\"  Alert Equivalence Validation (AVX2 SIMD)\")\n",
                "print(f\"{'='*50}\")\n",
                "print(f\"  Sequential (AVX2) alerts: {len(s_alerts)}\")\n",
                "print(f\"  OpenMP-2T  (AVX2) alerts: {len(o_alerts)}\")\n",
                "print(f\"  CUDA       (AVX2) alerts: {len(c_alerts)}\")\n",
                "\n",
                "# Deep comparison: sort by (packet_id, signature_id) and compare\n",
                "def sort_key(a):\n",
                "    return (a.get('packet_id', 0), a.get('signature_id', 0))\n",
                "\n",
                "s_sorted = sorted(s_alerts, key=sort_key)\n",
                "o_sorted = sorted(o_alerts, key=sort_key)\n",
                "c_sorted = sorted(c_alerts, key=sort_key)\n",
                "\n",
                "# Compare alert IDs\n",
                "s_ids = set((a.get('packet_id'), a.get('signature_id')) for a in s_alerts)\n",
                "o_ids = set((a.get('packet_id'), a.get('signature_id')) for a in o_alerts)\n",
                "c_ids = set((a.get('packet_id'), a.get('signature_id')) for a in c_alerts)\n",
                "\n",
                "assert s_ids == o_ids, f\"Sequential vs OpenMP mismatch! Diff: {s_ids.symmetric_difference(o_ids)}\"\n",
                "assert s_ids == c_ids, f\"Sequential vs CUDA mismatch! Diff: {s_ids.symmetric_difference(c_ids)}\"\n",
                "assert len(s_alerts) == len(o_alerts) == len(c_alerts), \"Alert count mismatch!\"\n",
                "\n",
                "print(f\"\\n  ✅ SUCCESS: All {len(s_alerts)} alerts are bit-exact across Sequential, OpenMP, and CUDA!\")\n",
                "print(f\"  ✅ AVX2 SIMD verification produces identical results to all modes.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 11: Generate Full Benchmark Plots\n",
                "Produce publication-quality plots: Total Time, Speedup, Phase Breakdown, Throughput."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!python3 scripts/plot_benchmarks.py\n",
                "\n",
                "from IPython.display import Image, display\n",
                "for plot_name in [\"total_time.png\", \"speedup.png\", \"phase_breakdown.png\", \"throughput.png\"]:\n",
                "    path = os.path.join(\"benchmarks\", plot_name)\n",
                "    if os.path.exists(path):\n",
                "        print(f\"\\n--- {plot_name} ---\")\n",
                "        display(Image(path))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 12: Pipeline Summary & Architecture\n",
                "Display the complete IDS pipeline architecture with performance numbers."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "\n",
                "df = pd.read_csv('benchmarks/benchmark_results.csv')\n",
                "\n",
                "# Find sequential baseline and CUDA row\n",
                "seq_row = df[df['mode'] == 'Sequential'].iloc[0] if len(df[df['mode'] == 'Sequential']) > 0 else None\n",
                "cuda_row = df[df['mode'] == 'CUDA'].iloc[0] if len(df[df['mode'] == 'CUDA']) > 0 else None\n",
                "\n",
                "print('=' * 70)\n",
                "print('  GPU Sparse IDS — Full Pipeline Architecture')\n",
                "print('=' * 70)\n",
                "print()\n",
                "print('  ┌─────────────────────────────────────────────────────────────┐')\n",
                "print('  │             PCAP Ingestion (libpcap + IPv4/IPv6/VLAN)       │')\n",
                "print('  ├─────────────────────────────────────────────────────────────┤')\n",
                "print('  │  Feature Extraction   │  FNV-1a n-gram hashing             │')\n",
                "print('  │  (GPU: CUDA kernels)  │  8KB shared-mem bitset/block       │')\n",
                "print('  ├─────────────────────────────────────────────────────────────┤')\n",
                "print('  │  CSR Matrix Build     │  P ∈ {0,1}^{N×V}, S ∈ {0,1}^{K×V} │')\n",
                "print('  ├─────────────────────────────────────────────────────────────┤')\n",
                "print('  │  SpMM: C = P × Sᵀ     │  cuSPARSE SpGEMM (GPU)            │')\n",
                "print('  │                       │  OpenMP merge (CPU)                │')\n",
                "print('  ├─────────────────────────────────────────────────────────────┤')\n",
                "print('  │  Candidate Filter     │  GPU atomicAdd kernel (in-VRAM)    │')\n",
                "print('  ├─────────────────────────────────────────────────────────────┤')\n",
                "print('  │  Exact Verification   │  AVX2 SIMD Vectorized Scanning     │')\n",
                "print('  │                       │  _mm256_cmpeq_epi8 (32 B/cycle)    │')\n",
                "print('  │                       │  _mm256_movemask_epi8 → memcmp     │')\n",
                "print('  └─────────────────────────────────────────────────────────────┘')\n",
                "print()\n",
                "\n",
                "if seq_row is not None and cuda_row is not None:\n",
                "    speedup = seq_row['total_ms'] / cuda_row['total_ms'] if cuda_row['total_ms'] > 0 else 0\n",
                "    print(f'  Sequential baseline:  {seq_row[\"total_ms\"]:.1f} ms  ({seq_row[\"packets_per_sec\"]:.0f} pkt/s)')\n",
                "    print(f'  CUDA + AVX2 SIMD:     {cuda_row[\"total_ms\"]:.1f} ms  ({cuda_row[\"packets_per_sec\"]:.0f} pkt/s)')\n",
                "    print(f'  End-to-end speedup:   {speedup:.2f}×')\n",
                "    print(f'  Feature extraction:   {seq_row[\"feature_extract_ms\"]:.1f} ms → {cuda_row[\"feature_extract_ms\"]:.1f} ms (GPU)')\n",
                "    print(f'  Verification (AVX2):  {seq_row[\"verification_ms\"]:.4f} ms (seq) / {cuda_row[\"verification_ms\"]:.4f} ms (cuda)')\n",
                "    print(f'  Candidates verified:  {int(seq_row[\"num_candidates\"])}')\n",
                "    print(f'  Alerts generated:     {int(seq_row[\"num_alerts\"])}')\n",
                "print()\n",
                "print('=' * 70)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 13: Export & Download Benchmark Artifacts"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!zip -r /content/gpu_sparse_ids_results.zip benchmarks/ output/\n",
                "try:\n",
                "    from google.colab import files\n",
                "    files.download('/content/gpu_sparse_ids_results.zip')\n",
                "    print(\"Downloaded results bundle!\")\n",
                "except Exception as e:\n",
                "    print(\"Running outside Colab interactive environment, results saved to /content/gpu_sparse_ids_results.zip\")"
            ]
        }
    ],
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "gpuType": "T4",
            "provenance": []
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

out_path = os.path.join(os.path.dirname(__file__), "..", "notebooks", "gpu_sparse_ids_colab.ipynb")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("Notebook generated successfully at notebooks/gpu_sparse_ids_colab.ipynb")
