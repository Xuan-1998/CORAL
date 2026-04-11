#!/usr/bin/env python3
"""Analyze multiagent gain sources through the RLVR lens.

Maps concepts from "Limit of RLVR" (Yue et al. 2025) to multiagent systems:
  - Sampling efficiency: does multiagent improve score per eval?
  - Capability boundary: does multiagent solve problems single-agent can't?
  - Exploration narrowing: does shared knowledge reduce diversity?
  - Knowledge as distillation: does shared knowledge introduce new capabilities?

Usage:
    python analyze_gains.py --results-dir ./results/ablation
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_attempts(run_dir: Path) -> list[dict]:
    """Load all attempt JSONs from a run's .coral/public/attempts/."""
    attempts_dir = run_dir / ".coral" / "public" / "attempts"
    if not attempts_dir.exists():
        return []
    attempts = []
    for f in sorted(attempts_dir.glob("*.json")):
        try:
            attempts.append(json.loads(f.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
    return attempts


def best_score_trajectory(attempts: list[dict], direction: str = "maximize") -> list[float]:
    """Compute running best score over attempts (sorted by timestamp)."""
    attempts = sorted(attempts, key=lambda a: a.get("timestamp", ""))
    best = None
    trajectory = []
    for a in attempts:
        s = a.get("score")
        if s is None:
            continue
        if best is None:
            best = s
        elif direction == "maximize" and s > best:
            best = s
        elif direction == "minimize" and s < best:
            best = s
        trajectory.append(best)
    return trajectory


def per_agent_attempts(attempts: list[dict]) -> dict[str, list[dict]]:
    """Group attempts by agent_id."""
    groups = defaultdict(list)
    for a in attempts:
        groups[a.get("agent_id", "unknown")].append(a)
    return dict(groups)


# ── Metric 1: Sampling Efficiency ──

def sampling_efficiency(attempts: list[dict], direction: str = "maximize") -> dict:
    """Compute improvement rate and score-per-eval metrics."""
    scored = [a for a in attempts if a.get("score") is not None]
    if not scored:
        return {"improvement_rate": 0, "total_evals": 0, "best_score": None}

    improved = [a for a in scored if a.get("status") == "improved"]
    best = max(scored, key=lambda a: a["score"]) if direction == "maximize" else min(scored, key=lambda a: a["score"])

    return {
        "improvement_rate": len(improved) / len(scored) if scored else 0,
        "total_evals": len(scored),
        "best_score": best["score"],
        "improved_count": len(improved),
    }


# ── Metric 2: Capability Boundary (pass@k analogy) ──

def capability_boundary(attempts: list[dict], thresholds: list[float]) -> dict[float, bool]:
    """For each score threshold, check if any attempt reached it.

    Analogous to pass@k: can the system solve problems at this difficulty?
    """
    scores = [a["score"] for a in attempts if a.get("score") is not None]
    if not scores:
        return {t: False for t in thresholds}
    max_score = max(scores)
    return {t: max_score >= t for t in thresholds}


def unique_solutions_reached(attempts: list[dict], bucket_size: float = 0.01) -> int:
    """Count distinct score buckets reached (solution diversity)."""
    scores = [a["score"] for a in attempts if a.get("score") is not None]
    buckets = set(round(s / bucket_size) for s in scores)
    return len(buckets)


# ── Metric 3: Exploration Diversity ──

def extract_strategy_keywords(title: str) -> set[str]:
    """Extract strategy keywords from attempt title."""
    if not title:
        return set()
    # Normalize and extract meaningful words
    words = re.findall(r'[a-z]+(?:-[a-z]+)*', title.lower())
    # Filter common stop words
    stop = {'the', 'a', 'an', 'to', 'and', 'or', 'with', 'for', 'in', 'on',
            'try', 'test', 'add', 'use', 'fix', 'update', 'change', 'set',
            'from', 'of', 'by', 'at', 'is', 'it', 'as', 'this', 'that'}
    return {w for w in words if w not in stop and len(w) > 2}


def exploration_diversity(attempts: list[dict]) -> dict:
    """Measure exploration diversity across agents.

    Returns pairwise Jaccard similarity of strategy vocabularies.
    High similarity = exploration narrowing (bad).
    Low similarity = diverse exploration (good).
    """
    agent_groups = per_agent_attempts(attempts)
    if len(agent_groups) < 2:
        return {"mean_jaccard": 0, "agent_count": len(agent_groups)}

    # Build per-agent strategy vocabulary
    vocabs = {}
    for aid, agent_attempts in agent_groups.items():
        vocab = set()
        for a in agent_attempts:
            vocab |= extract_strategy_keywords(a.get("title", ""))
        vocabs[aid] = vocab

    # Pairwise Jaccard
    agents = list(vocabs.keys())
    jaccards = []
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            a, b = vocabs[agents[i]], vocabs[agents[j]]
            if a | b:
                jaccards.append(len(a & b) / len(a | b))

    unique_per_agent = {aid: len(v - set.union(*(vocabs[a] for a in vocabs if a != aid)))
                        for aid, v in vocabs.items()}

    return {
        "mean_jaccard": float(np.mean(jaccards)) if jaccards else 0,
        "agent_vocabs": {aid: len(v) for aid, v in vocabs.items()},
        "unique_per_agent": unique_per_agent,
        "agent_count": len(agents),
    }


# ── Metric 4: Cross-Agent Transfer ──

def cross_agent_transfer(attempts: list[dict]) -> dict:
    """Measure how often agents build on each other's work."""
    agent_commits = defaultdict(set)
    for a in attempts:
        agent_commits[a.get("agent_id", "unknown")].add(a.get("commit_hash", ""))

    cross_parent = 0
    cross_improved = 0
    total = 0
    for a in attempts:
        aid = a.get("agent_id", "unknown")
        parent = a.get("parent_hash", "")
        if not parent:
            continue
        total += 1
        # Check if parent belongs to a different agent
        is_cross = any(parent in commits for other_aid, commits in agent_commits.items() if other_aid != aid)
        if is_cross:
            cross_parent += 1
            if a.get("status") == "improved":
                cross_improved += 1

    return {
        "cross_parent_rate": cross_parent / total if total else 0,
        "cross_improved": cross_improved,
        "cross_total": cross_parent,
        "total_with_parent": total,
    }


# ── Main Analysis ──

def analyze_run(run_dir: Path, direction: str = "maximize") -> dict:
    """Full analysis of a single run."""
    attempts = load_attempts(run_dir)
    if not attempts:
        return {"error": f"No attempts found in {run_dir}"}

    # Score thresholds for capability boundary (task-specific)
    scores = [a["score"] for a in attempts if a.get("score") is not None]
    if scores:
        score_range = np.linspace(min(scores), max(scores), 10)
    else:
        score_range = []

    return {
        "run_dir": str(run_dir),
        "total_attempts": len(attempts),
        "sampling_efficiency": sampling_efficiency(attempts, direction),
        "capability_boundary": {str(t): v for t, v in capability_boundary(attempts, list(score_range)).items()},
        "unique_solutions": unique_solutions_reached(attempts),
        "exploration_diversity": exploration_diversity(attempts),
        "cross_agent_transfer": cross_agent_transfer(attempts),
        "trajectory": best_score_trajectory(attempts, direction),
    }


def compare_conditions(results: dict[str, dict]) -> str:
    """Generate a comparison report across experimental conditions."""
    lines = ["# Multiagent Gain Source Analysis Report", ""]

    # ── Sampling Efficiency Comparison ──
    lines.append("## 1. Sampling Efficiency (RLVR analogy: pass@1 improvement)")
    lines.append("")
    lines.append("| Condition | Best Score | Improvement Rate | Total Evals |")
    lines.append("|-----------|-----------|-----------------|-------------|")
    for name, r in results.items():
        se = r.get("sampling_efficiency", {})
        lines.append(f"| {name} | {se.get('best_score', 'N/A')} | "
                     f"{se.get('improvement_rate', 0):.1%} | {se.get('total_evals', 0)} |")
    lines.append("")

    # ── Capability Boundary ──
    lines.append("## 2. Capability Boundary (RLVR analogy: pass@k at large k)")
    lines.append("")
    lines.append("Does multiagent reach scores that single-agent never reaches?")
    lines.append("")
    for name, r in results.items():
        lines.append(f"- **{name}**: {r.get('unique_solutions', 0)} unique score buckets reached")
    lines.append("")

    # ── Exploration Diversity ──
    lines.append("## 3. Exploration Diversity (RLVR analogy: exploration narrowing)")
    lines.append("")
    for name, r in results.items():
        ed = r.get("exploration_diversity", {})
        if ed.get("agent_count", 0) > 1:
            lines.append(f"- **{name}**: mean Jaccard = {ed.get('mean_jaccard', 0):.3f} "
                         f"(higher = more narrowing)")
    lines.append("")

    # ── Cross-Agent Transfer ──
    lines.append("## 4. Knowledge Transfer (RLVR analogy: distillation vs RL)")
    lines.append("")
    for name, r in results.items():
        ct = r.get("cross_agent_transfer", {})
        if ct.get("total_with_parent", 0) > 0:
            lines.append(f"- **{name}**: {ct.get('cross_parent_rate', 0):.1%} cross-agent parents, "
                         f"{ct.get('cross_improved', 0)} cross-agent improvements")
    lines.append("")

    # ── Key Findings ──
    lines.append("## 5. Key Findings")
    lines.append("")

    # Compare best-of-4-independent vs co-evolution
    coevol = results.get("4agent_coevol", {}).get("sampling_efficiency", {})
    indep = results.get("4agent_no_sharing", {}).get("sampling_efficiency", {})
    if coevol.get("best_score") and indep.get("best_score"):
        if coevol["best_score"] > indep["best_score"]:
            lines.append("- **Co-evolution > Independent**: Multiagent gain is NOT just sampling. "
                         "Shared knowledge genuinely helps.")
        else:
            lines.append("- **Co-evolution ≈ Independent**: Multiagent gain is mostly sampling efficiency. "
                         "Like RLVR, it doesn't expand the capability boundary.")

    attempts_only = results.get("4agent_attempts_only", {}).get("sampling_efficiency", {})
    full = results.get("4agent_coevol", {}).get("sampling_efficiency", {})
    if attempts_only.get("best_score") and full.get("best_score"):
        if full["best_score"] > attempts_only["best_score"] * 1.01:
            lines.append("- **Notes+Skills matter**: Knowledge sharing acts like distillation, "
                         "introducing capabilities beyond what attempt-sharing alone provides.")
        else:
            lines.append("- **Notes+Skills don't help much**: Knowledge sharing is not distillation. "
                         "Agents learn primarily from seeing each other's scored attempts.")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze multiagent gain sources")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--direction", default="maximize", choices=["maximize", "minimize"])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    # Discover runs
    results = {}
    for run_dir in sorted(args.results_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        # Look for .coral directory (could be nested under task slug / timestamp)
        coral_dirs = list(run_dir.rglob(".coral"))
        for cd in coral_dirs:
            rd = cd.parent
            name = rd.name
            print(f"Analyzing: {name} ({rd})")
            results[name] = analyze_run(rd, args.direction)

    if not results:
        print(f"No runs found in {args.results_dir}")
        return

    # Generate report
    report = compare_conditions(results)
    print(report)

    # Save
    out = args.output or args.results_dir / "analysis_report.md"
    out.write_text(report)
    print(f"\nReport saved to {out}")

    # Also save raw data
    raw_out = args.results_dir / "analysis_raw.json"
    # Remove non-serializable items
    for r in results.values():
        r.pop("trajectory", None)
    raw_out.write_text(json.dumps(results, indent=2, default=str))
    print(f"Raw data saved to {raw_out}")


if __name__ == "__main__":
    main()
