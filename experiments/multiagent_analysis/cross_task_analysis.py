#!/usr/bin/env python3
"""Cross-task analysis: maps RLVR claims to multiagent evidence across all tasks.

Produces the key tables/figures needed for a systematic RLVR-lens analysis:
1. pass@k curves (best_score@N) per task per condition
2. Capability boundary: which conditions reach which thresholds
3. Sampling efficiency gap
4. Exploration diversity
"""
import json, re, sys
from pathlib import Path
from collections import defaultdict

TASKS = {
    "circle_packing": {
        "dir": "results/ablation",
        "direction": "maximize",
        "benchmark": 1.0,
        "thresholds": [0.9, 0.95, 0.99, 0.999, 1.0],
    },
    "erdos": {
        "dir": "results/ablation_erdos",
        "direction": "maximize",
        "benchmark": 1.0,
        "thresholds": [0.9, 0.95, 0.99, 0.999, 1.0],
    },
    "kernel_builder": {
        "dir": "results/ablation_kernel",
        "direction": "minimize",
        "benchmark": 1363,
        "thresholds": [10000, 5000, 3000, 2000, 1500],
    },
    "heilbronn": {
        "dir": "results/ablation_heilbronn",
        "direction": "maximize",
        "benchmark": 1.0,
        "thresholds": [0.5, 0.7, 0.9, 0.95, 1.0],
    },
    "signal": {
        "dir": "results/ablation_signal",
        "direction": "maximize",
        "benchmark": 1.0,
        "thresholds": [0.5, 0.7, 0.9, 0.95, 1.0],
    },
}

CONDITIONS = ["1agent_baseline", "4agent_coevol", "4agent_attempts_only", "4agent_no_sharing"]

def load_attempts(base):
    attempts = []
    for f in Path(base).rglob("attempts/*.json"):
        try:
            d = json.loads(f.read_text())
            if d.get("score") is not None:
                attempts.append(d)
        except:
            pass
    attempts.sort(key=lambda a: a.get("timestamp", ""))
    return attempts

def is_better(a, b, direction):
    if direction == "maximize":
        return a > b
    return a < b

def best_at_k(attempts, direction):
    best = None
    curve = []
    for a in attempts:
        s = a["score"]
        if best is None or is_better(s, best, direction):
            best = s
        curve.append(best)
    return curve

def reached_threshold(curve, threshold, direction):
    for i, v in enumerate(curve):
        if direction == "maximize" and v >= threshold:
            return i + 1
        if direction == "minimize" and v <= threshold:
            return i + 1
    return None

sep = "=" * 80

print(sep)
print("CROSS-TASK ANALYSIS: RLVR LENS ON MULTIAGENT SYSTEMS")
print(f"Tasks: {len(TASKS)} | Conditions: {len(CONDITIONS)}")
print(sep)

# ============================================================
# CLAIM 1: "Base > RL at large pass@k"
# Analog: 1agent/no_sharing ≥ coevol at large N
# ============================================================
print(f"\n{'#'*60}")
print("# CLAIM 1: Capability Boundary")
print("# RLVR: 'Base model surpasses RL at large k'")
print("# Analog: Does 1agent/no_sharing beat coevol at large N?")
print(f"{'#'*60}")

for task_name, cfg in TASKS.items():
    base = cfg["dir"]
    direction = cfg["direction"]
    thresholds = cfg["thresholds"]

    data = {}
    for cond in CONDITIONS:
        d = Path(base) / cond
        if d.exists():
            atts = load_attempts(d)
            if atts:
                data[cond] = atts

    if not data:
        print(f"\n  {task_name}: no data yet")
        continue

    print(f"\n  === {task_name} (direction={direction}) ===")

    # Evals to reach threshold
    short = {"1agent_baseline": "1agent", "4agent_coevol": "coevol",
             "4agent_attempts_only": "att_only", "4agent_no_sharing": "no_share"}

    header = f"  {'Threshold':<12}"
    for cond in data:
        header += f" {short.get(cond, cond):>10}"
    print(header)
    print("  " + "-" * (12 + 11 * len(data)))

    for t in thresholds:
        row = f"  {t:<12}"
        for cond, atts in data.items():
            curve = best_at_k(atts, direction)
            k = reached_threshold(curve, t, direction)
            if k:
                row += f" {k:>10}"
            else:
                row += f" {'never':>10}"
        print(row)

    # Final scores
    print()
    header = f"  {'Metric':<12}"
    for cond in data:
        header += f" {short.get(cond, cond):>10}"
    print(header)
    print("  " + "-" * (12 + 11 * len(data)))

    row_n = f"  {'N evals':<12}"
    row_best = f"  {'Best':<12}"
    row_impr = f"  {'Impr%':<12}"
    for cond, atts in data.items():
        n = len(atts)
        curve = best_at_k(atts, direction)
        best = curve[-1] if curve else 0
        improved = sum(1 for a in atts if a.get("status") == "improved")
        rate = improved / n * 100 if n else 0
        row_n += f" {n:>10}"
        if direction == "minimize":
            row_best += f" {best:>10.0f}"
        else:
            row_best += f" {best:>10.6f}"
        row_impr += f" {rate:>9.0f}%"
    print(row_n)
    print(row_best)
    print(row_impr)

# ============================================================
# CLAIM 2: "RL boosts sampling efficiency but reduces boundary"
# ============================================================
print(f"\n{'#'*60}")
print("# CLAIM 2: Sampling Efficiency vs Boundary")
print("# RLVR: 'RL improves pass@1 but shrinks solution space'")
print("# Analog: coevol faster early but lower ceiling?")
print(f"{'#'*60}")

for task_name, cfg in TASKS.items():
    base = cfg["dir"]
    direction = cfg["direction"]

    data = {}
    for cond in CONDITIONS:
        d = Path(base) / cond
        if d.exists():
            atts = load_attempts(d)
            if atts:
                data[cond] = atts

    if len(data) < 2:
        continue

    print(f"\n  === {task_name} ===")

    # Compare at small N (first 5 evals) and large N (all evals)
    for label, max_k in [("Early (k=5)", 5), ("Mid (k=20)", 20), ("Late (all)", None)]:
        row = f"  {label:<16}"
        for cond, atts in data.items():
            subset = atts[:max_k] if max_k else atts
            if not subset:
                row += f" {'N/A':>10}"
                continue
            curve = best_at_k(subset, direction)
            best = curve[-1]
            if direction == "minimize":
                row += f" {best:>10.0f}"
            else:
                row += f" {best:>10.6f}"
        print(row)

# ============================================================
# CLAIM 3: Exploration Narrowing
# ============================================================
print(f"\n{'#'*60}")
print("# CLAIM 3: Exploration Narrowing")
print("# RLVR: 'RL narrows the solution space'")
print("# Analog: Does sharing reduce strategy diversity?")
print(f"{'#'*60}")

def keywords(title):
    if not title:
        return set()
    words = re.findall(r"[a-z]+(?:-[a-z]+)*", title.lower())
    stop = {"the","a","an","to","and","or","with","for","in","on","try","test",
            "add","use","fix","update","change","set","from","of","by","at","is",
            "it","as","this","that","initial","naive","baseline","eval","existing",
            "code","starting","provided"}
    return {w for w in words if w not in stop and len(w) > 2}

for task_name, cfg in TASKS.items():
    base = cfg["dir"]
    data = {}
    for cond in CONDITIONS:
        d = Path(base) / cond
        if d.exists():
            atts = load_attempts(d)
            if atts:
                data[cond] = atts

    if len(data) < 2:
        continue

    print(f"\n  === {task_name} ===")
    short = {"1agent_baseline": "1agent", "4agent_coevol": "coevol",
             "4agent_attempts_only": "att_only", "4agent_no_sharing": "no_share"}

    for cond, atts in data.items():
        agent_vocabs = defaultdict(set)
        for a in atts:
            agent_vocabs[a.get("agent_id", "?")].update(keywords(a.get("title", "")))

        if len(agent_vocabs) < 2:
            total_vocab = set().union(*agent_vocabs.values()) if agent_vocabs else set()
            print(f"  {short.get(cond, cond)}: 1 agent, {len(total_vocab)} unique terms")
            continue

        agents = list(agent_vocabs.keys())
        jaccards = []
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                a, b = agent_vocabs[agents[i]], agent_vocabs[agents[j]]
                if a | b:
                    jaccards.append(len(a & b) / len(a | b))
        mean_j = sum(jaccards) / len(jaccards) if jaccards else 0
        total_vocab = set().union(*agent_vocabs.values())
        print(f"  {short.get(cond, cond)}: Jaccard={mean_j:.3f}, {len(total_vocab)} unique terms")

# ============================================================
# CLAIM 4: Distillation ≠ RL
# ============================================================
print(f"\n{'#'*60}")
print("# CLAIM 4: Distillation vs RL")
print("# RLVR: 'Distillation expands boundary, RL does not'")
print("# Analog: attempts_only (score signal) vs coevol (full knowledge)")
print(f"{'#'*60}")

for task_name, cfg in TASKS.items():
    base = cfg["dir"]
    direction = cfg["direction"]

    coevol = load_attempts(str(Path(base) / "4agent_coevol"))
    att_only = load_attempts(str(Path(base) / "4agent_attempts_only"))

    if not coevol and not att_only:
        continue

    print(f"\n  === {task_name} ===")
    for label, atts in [("coevol (RL analog)", coevol), ("att_only (distill analog)", att_only)]:
        if not atts:
            print(f"  {label}: no data")
            continue
        curve = best_at_k(atts, direction)
        best = curve[-1]
        n = len(atts)
        improved = sum(1 for a in atts if a.get("status") == "improved")
        rate = improved / n * 100 if n else 0
        if direction == "minimize":
            print(f"  {label}: n={n}, best={best:.0f}, impr={rate:.0f}%")
        else:
            print(f"  {label}: n={n}, best={best:.8f}, impr={rate:.0f}%")

# ============================================================
# SUMMARY TABLE
# ============================================================
print(f"\n{'#'*60}")
print("# SUMMARY: Which condition wins on each task?")
print(f"{'#'*60}")

print(f"\n  {'Task':<20} {'Best Condition':<20} {'Best Score':<15} {'N evals':<10} {'2nd Best':<20}")
print("  " + "-" * 85)

for task_name, cfg in TASKS.items():
    base = cfg["dir"]
    direction = cfg["direction"]

    results = []
    for cond in CONDITIONS:
        d = Path(base) / cond
        if d.exists():
            atts = load_attempts(d)
            if atts:
                curve = best_at_k(atts, direction)
                results.append((cond, curve[-1], len(atts)))

    if not results:
        print(f"  {task_name:<20} no data")
        continue

    if direction == "maximize":
        results.sort(key=lambda x: x[1], reverse=True)
    else:
        results.sort(key=lambda x: x[1])

    short = {"1agent_baseline": "1agent", "4agent_coevol": "coevol",
             "4agent_attempts_only": "att_only", "4agent_no_sharing": "no_share"}

    best = results[0]
    second = results[1] if len(results) > 1 else ("N/A", 0, 0)

    if direction == "minimize":
        print(f"  {task_name:<20} {short.get(best[0], best[0]):<20} {best[1]:<15.0f} {best[2]:<10} {short.get(second[0], second[0]):<20}")
    else:
        print(f"  {task_name:<20} {short.get(best[0], best[0]):<20} {best[1]:<15.8f} {best[2]:<10} {short.get(second[0], second[0]):<20}")
