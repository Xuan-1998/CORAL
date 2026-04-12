#!/usr/bin/env python3
"""Deep mechanistic analysis: WHY does sharing hurt?

Three hypotheses:
1. Attention tax: agents spend turns reading shared state instead of coding
2. Strategy anchoring: agents copy strategies instead of exploring independently
3. Eval overhead: shared knowledge causes agents to attempt similar solutions,
   wasting evals on redundant approaches

We measure each by analyzing agent behavior traces.
"""
import json, re
from pathlib import Path
from collections import defaultdict
from datetime import datetime

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

def parse_ts(ts):
    try: return datetime.fromisoformat(ts)
    except: return None

sep = "=" * 70

for task, base_dir in [("Circle Packing", "results/ablation"), ("Erdos", "results/ablation_erdos")]:
    print(f"\n{sep}")
    print(f"MECHANISTIC ANALYSIS: {task}")
    print(sep)

    conditions = {}
    for cond in ["1agent_baseline", "4agent_coevol", "4agent_attempts_only", "4agent_no_sharing"]:
        d = Path(base_dir) / cond
        if d.exists():
            conditions[cond] = load_attempts(d)

    # === 1. EVAL VELOCITY: how fast do agents produce evals? ===
    print("\n## 1. Eval Velocity (evals per hour)")
    for name, atts in conditions.items():
        if len(atts) < 2: continue
        ts = [parse_ts(a["timestamp"]) for a in atts]
        ts = [t for t in ts if t]
        if len(ts) < 2: continue
        duration_h = (ts[-1] - ts[0]).total_seconds() / 3600
        if duration_h > 0:
            velocity = len(atts) / duration_h
            # Per-agent velocity
            agent_counts = defaultdict(int)
            for a in atts:
                agent_counts[a.get("agent_id", "?")] += 1
            n_agents = len(agent_counts)
            per_agent = velocity / n_agents if n_agents else velocity
            print(f"  {name}: {velocity:.1f} evals/h total, {per_agent:.1f}/agent/h ({n_agents} agents, {duration_h:.1f}h)")

    # === 2. REDUNDANCY: how often do agents try the same approach? ===
    print("\n## 2. Strategy Redundancy")
    def title_signature(title):
        """Extract a rough strategy signature from title."""
        if not title: return ""
        t = title.lower()
        # Extract key technique words
        techs = []
        for kw in ["l-bfgs", "adam", "annealing", "basin-hopping", "perturbation",
                    "multi-start", "multi-res", "hex", "grid", "inflate", "polish",
                    "penalty", "gradient", "haugland", "alphaevolve", "sigmoid",
                    "fft", "symmetric", "phase"]:
            if kw in t: techs.append(kw)
        return "|".join(sorted(techs))

    for name, atts in conditions.items():
        if not atts: continue
        sigs = [title_signature(a.get("title", "")) for a in atts]
        unique_sigs = len(set(s for s in sigs if s))
        total = len([s for s in sigs if s])
        redundancy = 1 - (unique_sigs / total) if total else 0

        # Per-agent: how many of agent's strategies were already tried by another?
        agent_sigs = defaultdict(list)
        for a in atts:
            agent_sigs[a.get("agent_id", "?")].append(title_signature(a.get("title", "")))

        if len(agent_sigs) > 1:
            duplicated = 0
            total_checked = 0
            for aid, my_sigs in agent_sigs.items():
                others = set()
                for oid, their_sigs in agent_sigs.items():
                    if oid != aid:
                        others.update(their_sigs)
                for s in my_sigs:
                    if s and s in others:
                        duplicated += 1
                    if s:
                        total_checked += 1
            dup_rate = duplicated / total_checked if total_checked else 0
            print(f"  {name}: {unique_sigs}/{total} unique strategies ({redundancy:.0%} redundancy), {dup_rate:.0%} cross-agent duplication")
        else:
            print(f"  {name}: {unique_sigs}/{total} unique strategies ({redundancy:.0%} redundancy)")

    # === 3. SCORE IMPROVEMENT PER EVAL (marginal value) ===
    print("\n## 3. Marginal Value of Each Eval")
    for name, atts in conditions.items():
        if not atts: continue
        best = 0
        improvements = []
        for a in atts:
            s = a["score"]
            if s > best:
                delta = s - best
                improvements.append(delta)
                best = s
        n_improved = len(improvements)
        n_total = len(atts)
        avg_delta = sum(improvements) / n_improved if n_improved else 0
        wasted = n_total - n_improved
        print(f"  {name}: {n_improved}/{n_total} evals improved ({wasted} wasted), avg improvement={avg_delta:.8f}")

    # === 4. TIME TO FIRST GOOD EVAL ===
    print("\n## 4. Time to First 'Good' Eval (score > 0.99)")
    for name, atts in conditions.items():
        if not atts: continue
        ts = [parse_ts(a["timestamp"]) for a in atts]
        first_ts = ts[0] if ts and ts[0] else None
        good_idx = None
        for i, a in enumerate(atts):
            if a["score"] >= 0.99:
                good_idx = i
                break
        if good_idx is not None and first_ts and ts[good_idx]:
            time_to_good = (ts[good_idx] - first_ts).total_seconds() / 60
            print(f"  {name}: eval #{good_idx+1}, {time_to_good:.0f} min after first eval")
        else:
            print(f"  {name}: never reached 0.99")

    # === 5. PLATEAU ANALYSIS: when do agents stop improving? ===
    print("\n## 5. Plateau Detection")
    for name, atts in conditions.items():
        if not atts: continue
        best = 0
        last_improvement_idx = 0
        for i, a in enumerate(atts):
            if a["score"] > best:
                best = a["score"]
                last_improvement_idx = i
        plateau_length = len(atts) - last_improvement_idx - 1
        print(f"  {name}: last improvement at eval #{last_improvement_idx+1}/{len(atts)}, plateau={plateau_length} evals ({plateau_length/len(atts)*100:.0f}%)")

    # === 6. CROSS-AGENT INFLUENCE QUALITY ===
    print("\n## 6. Cross-Agent Influence (coevol only)")
    coevol = conditions.get("4agent_coevol", [])
    if coevol:
        agent_commits = defaultdict(set)
        for a in coevol:
            agent_commits[a.get("agent_id", "?")].add(a.get("commit_hash", ""))

        cross_attempts = []
        self_attempts = []
        for a in coevol:
            aid = a.get("agent_id", "?")
            parent = a.get("parent_hash", "")
            if not parent: continue
            is_cross = any(parent in c for oid, c in agent_commits.items() if oid != aid)
            if is_cross:
                cross_attempts.append(a)
            else:
                self_attempts.append(a)

        cross_improved = sum(1 for a in cross_attempts if a.get("status") == "improved")
        self_improved = sum(1 for a in self_attempts if a.get("status") == "improved")
        cross_rate = cross_improved / len(cross_attempts) * 100 if cross_attempts else 0
        self_rate = self_improved / len(self_attempts) * 100 if self_attempts else 0

        cross_scores = [a["score"] for a in cross_attempts if a.get("status") == "improved"]
        self_scores = [a["score"] for a in self_attempts if a.get("status") == "improved"]
        cross_avg = sum(cross_scores) / len(cross_scores) if cross_scores else 0
        self_avg = sum(self_scores) / len(self_scores) if self_scores else 0

        print(f"  Cross-agent attempts: {len(cross_attempts)}, improved {cross_rate:.0f}%, avg score {cross_avg:.6f}")
        print(f"  Self-parent attempts: {len(self_attempts)}, improved {self_rate:.0f}%, avg score {self_avg:.6f}")
        if cross_rate < self_rate:
            print(f"  => Cross-agent transfer is LESS effective than self-improvement")
        else:
            print(f"  => Cross-agent transfer is MORE effective than self-improvement")
