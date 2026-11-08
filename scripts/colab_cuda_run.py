#!/usr/bin/env python3
# =============================================================
# GPU Sparse IDS — Google Colab / Kaggle CUDA Execution Script
# =============================================================
# This notebook-compatible script clones the repository,
# builds with CUDA enabled, and runs GPU benchmarks on a
# free NVIDIA T4/V100 instance.
#
# USAGE:
#   1. Open Google Colab: https://colab.research.google.com
#   2. Set runtime: Runtime > Change runtime type > GPU (T4)
#   3. Upload this file or paste into a cell with %%writefile
#   4. Run: !python3 colab_cuda_run.py
# =============================================================

import subprocess, sys, os

def run(cmd, check=True):
    print(f"\n$ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    if check and result.returncode != 0:
        print(f"FAILED (exit {result.returncode})")
        sys.exit(result.returncode)
    return result.returncode

print("="*60)
print(" GPU Sparse IDS — CUDA Build & Test on Colab")
print("="*60)

# 1. Check GPU
print("\n--- GPU Info ---")
run("nvidia-smi", check=False)

# 2. Install dependencies
print("\n--- Installing dependencies ---")
run("apt-get install -y -q libpcap-dev cmake build-essential")

# 3. Clone or use existing repo
repo_dir = "/content/gpu_sparse_ids"
if not os.path.exists(repo_dir):
    # Replace with your actual repo URL:
    run(f"git clone https://github.com/YOUR_USERNAME/gpu_sparse_ids.git {repo_dir}")
else:
    print(f"Using existing repo at {repo_dir}")
    run(f"cd {repo_dir} && git pull", check=False)

os.chdir(repo_dir)

# 4. Build with CUDA
print("\n--- Building with CUDA ---")
build_dir = f"{repo_dir}/build_cuda"
os.makedirs(build_dir, exist_ok=True)
run(f"cmake -B {build_dir} -S {repo_dir} "
    f"-DCMAKE_BUILD_TYPE=Release "
    f"-DUSE_CUDA=ON "
    f"-DCMAKE_CUDA_ARCHITECTURES=75")   # T4 = sm_75; V100 = sm_70
run(f"cmake --build {build_dir} -j4")

# 5. Generate test data
print("\n--- Generating test data ---")
os.makedirs(f"{repo_dir}/data/pcap", exist_ok=True)
os.makedirs(f"{repo_dir}/data/signatures", exist_ok=True)
os.makedirs(f"{repo_dir}/benchmarks", exist_ok=True)
run(f"{build_dir}/generate_test_data {repo_dir}/data")

# 6. Run test suite (sequential + OpenMP)
print("\n--- Running test suite ---")
run(f"{build_dir}/test_ids")

# 7. Run main IDS in sequential mode
print("\n--- Sequential IDS ---")
run(f"{build_dir}/sparse_ids "
    f"{repo_dir}/data/pcap/test_traffic.pcap "
    f"{repo_dir}/data/signatures/test_signatures.txt seq")

# 8. Run main IDS in CUDA mode
print("\n--- CUDA IDS ---")
run(f"{build_dir}/sparse_ids "
    f"{repo_dir}/data/pcap/test_traffic.pcap "
    f"{repo_dir}/data/signatures/test_signatures.txt cuda")

# 9. Run benchmark
print("\n--- Benchmark (Sequential + OpenMP + CUDA) ---")
run(f"{build_dir}/benchmark_ids "
    f"{repo_dir}/data/pcap/benchmark_traffic.pcap "
    f"{repo_dir}/data/signatures/test_signatures.txt "
    f"{repo_dir}/benchmarks/colab_benchmark_results.csv")

# 10. Plot results
print("\n--- Plotting ---")
run("pip install -q matplotlib numpy")
run(f"python3 {repo_dir}/scripts/plot_benchmarks.py "
    f"{repo_dir}/benchmarks/colab_benchmark_results.csv")

print("\n" + "="*60)
print(" CUDA run complete. Check benchmarks/ for results.")
print("="*60)

