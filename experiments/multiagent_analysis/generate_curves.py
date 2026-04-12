#!/usr/bin/env python3
"""Generate pass@k style curves (CSV) for plotting.

Outputs CSV files that can be plotted to show:
- x-axis: number of evals (k)
- y-axis: best score achieved (analogous to pass@k)

This directly mirrors the RLVR paper's core figure.
"""
import json, csv
from pathlib import Path

def load_sorted_attempts(base_dir):
    attempts = []
    for f in Path(base_dir).rglob("attempts/*.json"):
        try:
            d = json.loads(f.read_text())
            if d.get("score") is not None:
                attempts.append(d)
        except:
            pass
    attempts.sort(key=lambda a: a.get("timestamp", ""))
    return attempts

def best_at_k(attempts):
    """Return list of (k, best_score) tuples."""
    best = None
    curve = []
    for i, a in enumerate(attempts):
        s = a["score"]
        if best is None or s > best:
            best = s
        curve.append((i + 1, best))
    return curve

tasks = {
    "circle_packing": {
        "1agent": "results/ablation/1agent_baseline",
        "4agent_coevol": "results/ablation/4agent_coevol",
        "4agent_attempts_only": "results/ablation/4agent_attempts_only",
        "4agent_no_sharing": "results/ablation/4agent_no_sharing",
    },
    "erdos": {
        "1agent": "results/ablation_erdos/1agent_baseline",
        "4agent_coevol": "results/ablation_erdos/4agent_coevol",
        "4agent_attempts_only": "results/ablation_erdos/4agent_attempts_only",
        "4agent_no_sharing": "results/ablation_erdos/4agent_no_sharing",
    },
    "kernel_builder": {
        "1agent": "results/ablation_kernel/1agent_baseline",
        "4agent_coevol": "results/ablation_kernel/4agent_coevol",
        "4agent_no_sharing": "results/ablation_kernel/4agent_no_sharing",
    },
    "heilbronn": {
        "1agent": "results/ablation_heilbronn/1agent_baseline",
        "4agent_coevol": "results/ablation_heilbronn/4agent_coevol",
        "4agent_attempts_only": "results/ablation_heilbronn/4agent_attempts_only",
        "4agent_no_sharing": "results/ablation_heilbronn/4agent_no_sharing",
    },
    "signal": {
        "1agent": "results/ablation_signal/1agent_baseline",
        "4agent_coevol": "results/ablation_signal/4agent_coevol",
        "4agent_attempts_only": "results/ablation_signal/4agent_attempts_only",
        "4agent_no_sharing": "results/ablation_signal/4agent_no_sharing",
    },
    "hexagon_12": {
        "4agent_coevol": "results/ablation_hexagon12/4agent_coevol",
        "4agent_no_sharing": "results/ablation_hexagon12/4agent_no_sharing",
    },
    "matmul": {
        "1agent": "results/ablation_matmul/1agent_baseline",
        "4agent_coevol": "results/ablation_matmul/4agent_coevol",
        "4agent_no_sharing": "results/ablation_matmul/4agent_no_sharing",
    },
}

for task_name, conditions in tasks.items():
    outfile = f"results/pass_at_k_{task_name}.csv"
    
    # Collect all curves
    curves = {}
    max_k = 0
    for cond_name, base_dir in conditions.items():
        atts = load_sorted_attempts(base_dir)
        if atts:
            curves[cond_name] = best_at_k(atts)
            max_k = max(max_k, len(atts))
    
    if not curves:
        continue
    
    # Write CSV
    with open(outfile, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["k"] + list(curves.keys())
        writer.writerow(header)
        
        for k in range(1, max_k + 1):
            row = [k]
            for cond_name in curves:
                curve = curves[cond_name]
                if k <= len(curve):
                    row.append(f"{curve[k-1][1]:.10f}")
                else:
                    # Repeat last value
                    row.append(f"{curve[-1][1]:.10f}")
            writer.writerow(row)
    
    print(f"Saved {outfile} ({max_k} rows, {len(curves)} conditions)")
    
    # Also print summary
    print(f"\n  {task_name} pass@k summary:")
    for cond_name, curve in curves.items():
        n = len(curve)
        final = curve[-1][1]
        # Find k where score first exceeds 0.99
        k99 = next((k for k, s in curve if s >= 0.99), "never")
        k999 = next((k for k, s in curve if s >= 0.999), "never")
        print(f"    {cond_name}: n={n}, final={final:.8f}, k@0.99={k99}, k@0.999={k999}")
    print()
