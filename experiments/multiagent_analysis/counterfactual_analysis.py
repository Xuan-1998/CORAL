#!/usr/bin/env python3
"""Counterfactual analysis: what if we could cherry-pick the best of each condition?

Simulates hybrid strategies by combining attempts from different conditions:
1. "Oracle best-of-4": best score at each k from 4 independent 1-agent runs
2. "Selective sharing": only share when cross-agent transfer would improve
3. "Late sharing": independent early, shared late

This helps design better sharing mechanisms.
"""
import json
from pathlib import Path
from collections import defaultdict

def load_attempts(base):
    attempts = []
    for f in Path(base).rglob("attempts/*.json"):
        try:
            d = json.loads(f.read_text())
            if d.get("score") is not None:
                attempts.append(d)
        except: pass
    attempts.sort(key=lambda a: a.get("timestamp", ""))
    return attempts

def best_at_k(attempts, max_k=None):
    best = 0
    curve = []
    for i, a in enumerate(attempts):
        best = max(best, a["score"])
        curve.append(best)
        if max_k and i + 1 >= max_k:
            break
    return curve

sep = "=" * 70

for task, base_dir in [("Circle Packing", "results/ablation"), ("Erdos", "results/ablation_erdos")]:
    print(f"\n{sep}")
    print(f"COUNTERFACTUAL ANALYSIS: {task}")
    print(sep)

    conditions = {}
    for cond in ["1agent_baseline", "4agent_coevol", "4agent_attempts_only", "4agent_no_sharing"]:
        d = Path(base_dir) / cond
        if d.exists():
            conditions[cond] = load_attempts(d)

    if not conditions: continue

    # === Counterfactual 1: Oracle best-of-4 independent agents ===
    # Simulate: what if we ran 4 independent 1-agent runs and took the best?
    print("\n## Counterfactual 1: Oracle Best-of-4 Independent")
    single = conditions.get("1agent_baseline", [])
    if single:
        # Split single agent's attempts into 4 "virtual agents" by cycling
        virtual_agents = [[], [], [], []]
        for i, a in enumerate(single):
            virtual_agents[i % 4].append(a)

        # Best-of-4 at each k
        max_k = max(len(va) for va in virtual_agents)
        oracle_curve = []
        for k in range(max_k):
            best = 0
            for va in virtual_agents:
                if k < len(va):
                    best = max(best, va[k]["score"])
            oracle_curve.append(best if best > (oracle_curve[-1] if oracle_curve else 0) else (oracle_curve[-1] if oracle_curve else 0))

        print(f"  1agent has {len(single)} evals")
        print(f"  Simulated 4 virtual agents with {[len(va) for va in virtual_agents]} evals each")
        if oracle_curve:
            print(f"  Oracle best-of-4 final: {oracle_curve[-1]:.10f}")

    # === Counterfactual 2: Per-agent best scores ===
    print("\n## Counterfactual 2: Per-Agent Best Scores")
    for name, atts in conditions.items():
        agent_bests = defaultdict(float)
        for a in atts:
            aid = a.get("agent_id", "?")
            agent_bests[aid] = max(agent_bests[aid], a["score"])
        overall_best = max(agent_bests.values()) if agent_bests else 0
        best_agent = max(agent_bests, key=agent_bests.get) if agent_bests else "?"
        print(f"  {name}:")
        for aid in sorted(agent_bests):
            marker = " <-- best" if aid == best_agent else ""
            print(f"    {aid}: {agent_bests[aid]:.10f}{marker}")

    # === Counterfactual 3: What if coevol agents had no_sharing's velocity? ===
    print("\n## Counterfactual 3: Efficiency-Adjusted Comparison")
    print("  (What if all conditions had the same number of evals?)")
    min_n = min(len(atts) for atts in conditions.values() if atts)
    print(f"  Normalizing to first {min_n} evals:")
    for name, atts in conditions.items():
        if not atts: continue
        curve = best_at_k(atts, min_n)
        if curve:
            print(f"    {name}: best@{min_n} = {curve[-1]:.10f}")

    # === Counterfactual 4: Late sharing simulation ===
    print("\n## Counterfactual 4: Phase Analysis (Early vs Late)")
    for name, atts in conditions.items():
        if not atts: continue
        n = len(atts)
        mid = n // 2
        early = atts[:mid]
        late = atts[mid:]

        early_improved = sum(1 for a in early if a.get("status") == "improved")
        late_improved = sum(1 for a in late if a.get("status") == "improved")
        early_rate = early_improved / len(early) * 100 if early else 0
        late_rate = late_improved / len(late) * 100 if late else 0

        early_best = max((a["score"] for a in early), default=0)
        late_best = max((a["score"] for a in late), default=0)

        print(f"  {name}:")
        print(f"    Early (1-{mid}): {early_rate:.0f}% improved, best={early_best:.8f}")
        print(f"    Late ({mid+1}-{n}): {late_rate:.0f}% improved, best={late_best:.8f}")

    # === Key insight: compute-normalized comparison ===
    print("\n## Key Insight: Compute-Normalized Comparison")
    print("  (Comparing at equal TOTAL compute, not equal evals)")
    print("  4 agents × T hours = 4T agent-hours")
    print("  1 agent × 4T hours = 4T agent-hours")
    print()
    single_atts = conditions.get("1agent_baseline", [])
    coevol_atts = conditions.get("4agent_coevol", [])
    noshare_atts = conditions.get("4agent_no_sharing", [])

    if single_atts and coevol_atts:
        # At matched agent-hours: 1agent gets N evals, 4agent gets ~4N evals
        # But 4agent's per-agent velocity differs, so use actual counts
        single_best = max(a["score"] for a in single_atts)
        coevol_best = max(a["score"] for a in coevol_atts)
        noshare_best = max(a["score"] for a in noshare_atts) if noshare_atts else 0

        print(f"  1agent ({len(single_atts)} evals, 1 agent): {single_best:.10f}")
        print(f"  coevol ({len(coevol_atts)} evals, 4 agents): {coevol_best:.10f}")
        if noshare_atts:
            print(f"  no_sharing ({len(noshare_atts)} evals, 4 agents): {noshare_best:.10f}")

        # Compute ratio
        if coevol_best > 0:
            ratio = len(coevol_atts) / len(single_atts) if len(single_atts) > 0 else float('inf')
            print(f"\n  coevol uses {ratio:.1f}x more evals than 1agent")
            if coevol_best > single_best:
                print(f"  coevol score is {(coevol_best - single_best):.10f} higher")
                print(f"  => {ratio:.0f}x compute for {(coevol_best/single_best - 1)*100:.4f}% improvement")
            else:
                print(f"  coevol score is {(single_best - coevol_best):.10f} LOWER despite {ratio:.1f}x more evals")
                print(f"  => MORE compute, WORSE result")
