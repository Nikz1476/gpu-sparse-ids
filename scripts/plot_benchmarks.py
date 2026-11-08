#!/usr/bin/env python3
"""
GPU Sparse IDS — Benchmark Visualization
Generates performance plots from benchmark_results.csv
"""

import csv
import sys
import os

def read_csv(path):
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def to_f(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return 0.0

def plot_ascii_bar(title, labels, values, unit="ms", width=50):
    max_val = max(values) if values else 1
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    for label, val in zip(labels, values):
        bar_len = int((val / max_val) * width) if max_val > 0 else 0
        bar = "█" * bar_len
        print(f"  {label:<25} {bar:<50} {val:.2f} {unit}")
    print()

def main():
    csv_path = "benchmarks/benchmark_results.csv"
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]

    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found. Run benchmark_ids first.")
        sys.exit(1)

    rows = read_csv(csv_path)
    if not rows:
        print("ERROR: Empty CSV")
        sys.exit(1)

    print("\n" + "="*60)
    print("  GPU Sparse IDS — Benchmark Report")
    print("="*60)

    # Summary table
    print(f"\n{'Mode':<20} {'Threads':>7} {'Total(ms)':>10} {'Pkt/s':>10} {'Speedup':>8} {'Candidates':>11} {'Alerts':>7}")
    print("-"*80)

    seq_total = None
    for r in rows:
        mode = r['mode']
        threads = r['threads']
        total = to_f(r['total_ms'])
        candidates = int(to_f(r.get('num_candidates', 0)))
        alerts = int(to_f(r.get('num_alerts', 0)))
        speedup = to_f(r.get('speedup', 1.0))
        pps = to_f(r.get('packets_per_sec', 0))

        if seq_total is None and mode == 'Sequential':
            seq_total = total

        label = f"{mode}"
        if mode == 'OpenMP':
            label = f"OpenMP-{threads}T"

        print(f"  {label:<18} {threads:>7} {total:>10.1f} {pps:>10.0f} {speedup:>8.3f}x {candidates:>11} {alerts:>7}")

    # Phase breakdown
    print("\n" + "="*60)
    print("  Phase Breakdown (ms)")
    print("="*60)
    print(f"\n{'Mode':<20} {'FeatExt':>8} {'CSRBld':>8} {'SpMM':>8} {'Verify':>8} {'Total':>8}")
    print("-"*60)
    for r in rows:
        mode = r['mode']
        threads = r['threads']
        label = mode if mode != 'OpenMP' else f"OpenMP-{threads}T"
        fe  = to_f(r['feature_extract_ms'])
        csr = to_f(r['csr_construct_ms'])
        spm = to_f(r['sparse_compute_ms'])
        ver = to_f(r['verification_ms'])
        tot = to_f(r['total_ms'])
        print(f"  {label:<18} {fe:>8.1f} {csr:>8.1f} {spm:>8.1f} {ver:>8.3f} {tot:>8.1f}")

    # Matrix info (from first row)
    r0 = rows[0]
    print(f"\n{'='*60}")
    print("  Matrix Dimensions")
    print(f"{'='*60}")
    n_pkts = int(to_f(r0.get('num_packets', 0)))
    n_sigs = int(to_f(r0.get('num_signatures', 0)))
    vocab  = int(to_f(r0.get('vocab_size', 0)))
    nnz_p  = int(to_f(r0.get('nnz_packets', 0)))
    nnz_s  = int(to_f(r0.get('nnz_signatures', 0)))
    sparsity_p = 100.0 * (1.0 - nnz_p / (n_pkts * vocab)) if n_pkts * vocab > 0 else 0
    sparsity_s = 100.0 * (1.0 - nnz_s / (n_sigs * vocab)) if n_sigs * vocab > 0 else 0
    print(f"  P (packets×vocab):  {n_pkts} × {vocab}  nnz={nnz_p}  sparsity={sparsity_p:.2f}%")
    print(f"  S (sigs×vocab):     {n_sigs} × {vocab}  nnz={nnz_s}  sparsity={sparsity_s:.2f}%")
    print(f"  C (scores):         {n_pkts} × {n_sigs}")

    # ASCII speedup bar
    labels = []
    speedups = []
    for r in rows:
        mode = r['mode']
        t = r['threads']
        lbl = mode if mode != 'OpenMP' else f"OpenMP-{t}T"
        labels.append(lbl)
        speedups.append(to_f(r.get('speedup', 1.0)))

    plot_ascii_bar("Speedup vs Sequential", labels, speedups, unit="x")

    # ASCII total time bar
    times = [to_f(r['total_ms']) for r in rows]
    lbls  = []
    for r in rows:
        mode = r['mode']
        t = r['threads']
        lbls.append(mode if mode != 'OpenMP' else f"OpenMP-{t}T")

    plot_ascii_bar("Total Execution Time", lbls, times, unit="ms")

    # Try matplotlib if available
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np

        outdir = "benchmarks"
        os.makedirs(outdir, exist_ok=True)

        # --- Plot 1: Total time comparison ---
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = []
        for r in rows:
            m = r['mode']
            if m == 'Sequential':  colors.append('#2196F3')
            elif m == 'OpenMP':    colors.append('#4CAF50')
            else:                  colors.append('#FF9800')

        bars = ax.bar(lbls, times, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Execution Mode', fontsize=12)
        ax.set_ylabel('Total Time (ms)', fontsize=12)
        ax.set_title('GPU Sparse IDS — Total Execution Time', fontsize=14, fontweight='bold')
        for bar, t in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{t:.0f}ms', ha='center', va='bottom', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{outdir}/total_time.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {outdir}/total_time.png")

        # --- Plot 2: Speedup ---
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(lbls, speedups, color=colors, edgecolor='black', linewidth=0.5)
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Sequential baseline')
        ax.set_xlabel('Execution Mode', fontsize=12)
        ax.set_ylabel('Speedup', fontsize=12)
        ax.set_title('GPU Sparse IDS — Speedup vs Sequential', fontsize=14, fontweight='bold')
        for bar, s in zip(bars, speedups):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{s:.2f}x', ha='center', va='bottom', fontsize=9)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{outdir}/speedup.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {outdir}/speedup.png")

        # --- Plot 3: Phase breakdown stacked bar ---
        phases = {
            'Feature Extract': [to_f(r['feature_extract_ms']) for r in rows],
            'CSR Build':       [to_f(r['csr_construct_ms'])   for r in rows],
            'SpMM':            [to_f(r['sparse_compute_ms'])  for r in rows],
            'Verification':    [to_f(r['verification_ms'])    for r in rows],
        }
        phase_colors = ['#E91E63', '#9C27B0', '#FF5722', '#009688']
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(lbls))
        bottoms = np.zeros(len(lbls))
        for (phase_name, vals), col in zip(phases.items(), phase_colors):
            ax.bar(x, vals, bottom=bottoms, label=phase_name, color=col, edgecolor='black', linewidth=0.5)
            bottoms += np.array(vals)
        ax.set_xticks(x)
        ax.set_xticklabels(lbls, rotation=15, ha='right')
        ax.set_xlabel('Execution Mode', fontsize=12)
        ax.set_ylabel('Time (ms)', fontsize=12)
        ax.set_title('GPU Sparse IDS — Pipeline Phase Breakdown', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{outdir}/phase_breakdown.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {outdir}/phase_breakdown.png")

        # --- Plot 4: Packets per second ---
        pps_vals = [to_f(r.get('packets_per_sec', 0)) for r in rows]
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(lbls, pps_vals, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Execution Mode', fontsize=12)
        ax.set_ylabel('Packets / Second', fontsize=12)
        ax.set_title('GPU Sparse IDS — Throughput (Packets/sec)', fontsize=14, fontweight='bold')
        for bar, pps in zip(bars, pps_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                    f'{int(pps):,}', ha='center', va='bottom', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{outdir}/throughput.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {outdir}/throughput.png")

        print("\n  All plots saved to benchmarks/")

    except ImportError:
        print("\n  Note: matplotlib not available. Install with: pip install matplotlib")
        print("  ASCII summary printed above.")

    print("\n" + "="*60)
    print("  Report complete.")
    print("="*60 + "\n")

if __name__ == '__main__':
    main()

