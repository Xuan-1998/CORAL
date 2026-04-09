"""Evolutionary TTT: population-based search with crossover.

Maintains a population of solutions. Each step:
1. Select 2 parents (tournament selection by score)
2. LLM generates child by combining ideas from both parents
3. Evaluate child, add to population if good enough
4. Cull worst solutions to maintain population size

This is closer to AlphaEvolve/ThetaEvolve's actual mechanism.

Usage:
  python -m ttt.evolve \
    --task examples/kernel_engineering/trimul/task.yaml \
    --model Qwen/Qwen3-32B \
    --vllm-url http://localhost:8000/v1 \
    --steps 40 --population 8
"""

import argparse
import json
import logging
import os
import random
import re
import sys
from pathlib import Path

import requests
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)


def call_vllm(url, model, messages, max_tokens=16384, temperature=1.0):
    resp = requests.post(f"{url}/chat/completions", json={
        "model": model, "messages": messages,
        "max_tokens": max_tokens, "temperature": temperature,
    }, timeout=300)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def extract_code(response):
    if "</think>" in response:
        response = response[response.index("</think>") + 8:]
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
    return blocks[-1].strip() if blocks else ""


def run_grader(task_yaml, code, workdir):
    from ttt.pipeline import run_coral_grader
    return run_coral_grader(task_yaml, code, workdir)


SYSTEM = """/no_think You are an expert Triton kernel engineer.
You will be shown one or two parent solutions. Generate an IMPROVED child solution that:
- Combines the best ideas from the parent(s)
- Fixes any correctness issues
- Optimizes for speed

Output ONLY a complete Python file in ```python``` blocks. No explanations."""


def build_mutation_prompt(task_desc, parent, feedback):
    """Single parent → mutate."""
    msg = f"## Task\n{task_desc[:1500]}\n\n"
    msg += f"## Parent solution (score: {parent['score']:.4f})\n```python\n{parent['code'][:4000]}\n```\n\n"
    if feedback:
        msg += f"## Eval feedback\n{feedback[:500]}\n\n"
    msg += "Improve this solution. Fix bugs if score=0, optimize speed if score>0."
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": msg}]


def build_crossover_prompt(task_desc, parent1, parent2):
    """Two parents → crossover."""
    msg = f"## Task\n{task_desc[:1200]}\n\n"
    msg += f"## Parent A (score: {parent1['score']:.4f})\n```python\n{parent1['code'][:2500]}\n```\n\n"
    msg += f"## Parent B (score: {parent2['score']:.4f})\n```python\n{parent2['code'][:2500]}\n```\n\n"
    msg += "Combine the best ideas from BOTH parents into one improved solution."
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": msg}]


def tournament_select(population, k=3):
    """Select one individual via tournament selection."""
    candidates = random.sample(population, min(k, len(population)))
    return max(candidates, key=lambda x: x["score"])


def main():
    parser = argparse.ArgumentParser(description="Evolutionary TTT")
    parser.add_argument("--task", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-32B")
    parser.add_argument("--vllm-url", required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--crossover-rate", type=float, default=0.3)
    parser.add_argument("--workdir", default="/fsx/xuanj/ttt_evolve_results")
    args = parser.parse_args()

    with open(args.task) as f:
        task_cfg = yaml.safe_load(f)
    task_desc = task_cfg["task"]["description"]

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    seed_dir = Path(args.task).parent / "seed"
    if seed_dir.exists():
        import shutil
        for f in seed_dir.iterdir():
            if f.is_file():
                shutil.copy2(f, workdir)

    submission = workdir / "submission.py"
    seed_code = submission.read_text() if submission.exists() else ""

    # Initialize population with seed
    population = [{"code": seed_code, "score": 0.0, "feedback": "seed", "gen": 0}]
    best_score = 0.0
    best_code = seed_code
    total_evals = 0

    log.info("=== Evolutionary TTT: pop=%d, steps=%d, crossover=%.0f%% ===",
             args.population, args.steps, args.crossover_rate * 100)

    for step in range(args.steps):
        log.info("══════ Step %d/%d (pop=%d, best=%.4f) ══════",
                 step + 1, args.steps, len(population), best_score)

        # Decide: mutation or crossover
        if len(population) >= 2 and random.random() < args.crossover_rate:
            # Crossover
            p1 = tournament_select(population)
            p2 = tournament_select(population)
            while p2 is p1 and len(population) > 1:
                p2 = tournament_select(population)
            messages = build_crossover_prompt(task_desc, p1, p2)
            op = f"crossover({p1['score']:.4f}×{p2['score']:.4f})"
        else:
            # Mutation
            parent = tournament_select(population)
            messages = build_mutation_prompt(task_desc, parent, parent["feedback"])
            op = f"mutate({parent['score']:.4f})"

        try:
            response = call_vllm(args.vllm_url, args.model, messages)
            code = extract_code(response)
            if not code or len(code) < 50:
                log.warning("  [%s] No valid code", op)
                continue

            score, feedback = run_grader(args.task, code, workdir)
            total_evals += 1

            child = {"code": code, "score": score, "feedback": feedback, "gen": step + 1}
            log.info("  [%s] → score=%.4f %s", op, score, feedback[:100])

            # Add to population
            population.append(child)

            # Cull if over capacity (keep best + some diversity)
            if len(population) > args.population:
                population.sort(key=lambda x: x["score"], reverse=True)
                population = population[:args.population]

            if score > best_score:
                best_score = score
                best_code = code
                log.info("  ★ New best: %.4f (gen %d, eval #%d)", best_score, step + 1, total_evals)
                (workdir / "best_submission.py").write_text(best_code)

        except Exception as e:
            log.warning("  [%s] Failed: %s", op, e)

    # Summary
    log.info("\n=== Final Results ===")
    log.info("Best score: %.4f", best_score)
    log.info("Total evals: %d", total_evals)
    log.info("Population scores: %s", [f"{p['score']:.4f}" for p in sorted(population, key=lambda x: -x["score"])])
    (workdir / "best_submission.py").write_text(best_code)


if __name__ == "__main__":
    main()
