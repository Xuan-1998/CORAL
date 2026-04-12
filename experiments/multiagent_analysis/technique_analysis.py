#!/usr/bin/env python3
"""Analyze technique convergence across conditions."""
import json, re
from pathlib import Path
from collections import defaultdict, Counter

def extract_techniques(title):
    if not title: return set()
    t = title.lower()
    techniques = set()
    kw_map = {
        "l-bfgs": "L-BFGS-B", "lbfgs": "L-BFGS-B", "bfgs": "L-BFGS-B",
        "adam": "Adam", "sgd": "SGD",
        "simulated annealing": "SA", "annealing": "SA",
        "basin-hopping": "basin-hopping", "basin hopping": "basin-hopping",
        "perturbation": "perturbation",
        "multi-start": "multi-start", "multistart": "multi-start",
        "multi-res": "multi-resolution", "resolution": "multi-resolution",
        "sigmoid": "sigmoid-param",
        "fft": "FFT", "symmetric": "symmetric", "penalty": "penalty",
        "gradient": "gradient", "haugland": "Haugland-init",
        "alphaevolve": "AlphaEvolve-ref", "hex": "hex-grid",
        "grid": "grid-init", "inflate": "inflate-repair",
        "polish": "polish", "phase": "multi-phase",
    }
    for kw, tech in kw_map.items():
        if kw in t:
            techniques.add(tech)
    return techniques

sep = "=" * 60
for task, base in [("Circle Packing", "results/ablation"), ("Erdos", "results/ablation_erdos")]:
    print(f"\n{sep}")
    print(f"TECHNIQUE ANALYSIS: {task}")
    print(sep)

    for cond in ["4agent_coevol", "4agent_no_sharing"]:
        d = Path(base) / cond
        if not d.exists():
            continue

        agent_techs = defaultdict(Counter)
        for f in d.rglob("attempts/*.json"):
            try:
                data = json.loads(f.read_text())
                aid = data.get("agent_id", "?")
                techs = extract_techniques(data.get("title", ""))
                for t in techs:
                    agent_techs[aid][t] += 1
            except:
                pass

        print(f"\n--- {cond} ---")
        all_techs = set()
        for aid in sorted(agent_techs):
            techs = agent_techs[aid]
            all_techs.update(techs.keys())
            top = ", ".join(f"{t}({c})" for t, c in techs.most_common(5))
            print(f"  {aid}: {top}")

        agents = sorted(agent_techs.keys())
        if len(agents) >= 2:
            overlaps = []
            for i in range(len(agents)):
                for j in range(i + 1, len(agents)):
                    a = set(agent_techs[agents[i]].keys())
                    b = set(agent_techs[agents[j]].keys())
                    if a | b:
                        overlaps.append(len(a & b) / len(a | b))
            mean_overlap = sum(overlaps) / len(overlaps)
            print(f"  Technique overlap (Jaccard): {mean_overlap:.3f}")
            print(f"  Total unique techniques: {len(all_techs)}")
