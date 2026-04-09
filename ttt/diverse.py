"""Multi-Strategy TTT: diverse exploration with knowledge sharing.

Multiple "strategies" (different system prompts) explore in parallel.
After each round, the best code + learned insights are shared across all strategies.
Each strategy is a full solver, not a partial role.

Key difference from cooperative.py:
- No role splitting — every strategy is a complete solver
- Diversity comes from different optimization APPROACHES, not different ROLES
- Knowledge sharing via a shared "notebook" of what worked/failed

Usage:
  python -m ttt.diverse \
    --task examples/kernel_engineering/trimul/task.yaml \
    --model Qwen/Qwen3-32B \
    --vllm-url http://localhost:8000/v1 \
    --steps 20
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import requests
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)


def call_vllm(url, model, messages, max_tokens=16384, temperature=0.8):
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


# ── Diverse Strategies ────────────────────────────────────────────────

STRATEGIES = [
    {
        "name": "Fusion",
        "system": "/no_think You are a Triton kernel expert specializing in OPERATOR FUSION. "
                  "Your approach: fuse LayerNorm + projections + gating into minimal kernel launches. "
                  "Minimize memory traffic by keeping intermediate results in registers/shared memory. "
                  "Output ONLY complete Python in ```python``` blocks.",
    },
    {
        "name": "Precision",
        "system": "/no_think You are a Triton kernel expert specializing in MIXED PRECISION. "
                  "Your approach: use FP16/BF16 for compute-bound ops, FP32 for reductions. "
                  "Delegate the O(N^3) matmul to cuBLAS in FP16 for TensorCore utilization. "
                  "Output ONLY complete Python in ```python``` blocks.",
    },
    {
        "name": "Memory",
        "system": "/no_think You are a Triton kernel expert specializing in MEMORY OPTIMIZATION. "
                  "Your approach: optimize memory access patterns, use tiling to fit in L2 cache, "
                  "minimize global memory reads by recomputing cheap values. "
                  "Output ONLY complete Python in ```python``` blocks.",
    },
    {
        "name": "Hybrid",
        "system": "/no_think You are a Triton kernel expert. Use a HYBRID approach: "
                  "implement simple ops (LayerNorm, gating, sigmoid) as Triton kernels, "
                  "but delegate the expensive matmul to torch.bmm or cuBLAS. "
                  "Focus on correctness first, then optimize. "
                  "Output ONLY complete Python in ```python``` blocks.",
    },
]


def build_prompt(strategy, task_desc, parent_code, parent_score, feedback, notebook):
    """Build prompt with strategy-specific system + shared notebook."""
    msg = f"## Task\n{task_desc[:2000]}\n\n"
    msg += f"## Current best code (score: {parent_score:.4f})\n```python\n{parent_code[:3000]}\n```\n\n"
    if feedback:
        msg += f"## Last eval feedback\n{feedback[:500]}\n\n"

    # Shared notebook — key innovation: all strategies see what others learned
    if notebook:
        msg += "## Shared Notebook (insights from all strategies)\n"
        for entry in notebook[-10:]:
            msg += f"- [{entry['strategy']}] score={entry['score']:.4f}: {entry['insight']}\n"
        msg += "\n"

    msg += "IMPORTANT: If previous code failed correctness, fix that first. "
    msg += "Generate a COMPLETE working file.\n"

    return [{"role": "system", "content": strategy["system"]},
            {"role": "user", "content": msg}]


def extract_insight(code, score, feedback, strategy_name):
    """Auto-extract a brief insight from the attempt."""
    if score > 0:
        # Successful — note what approach worked
        if "cuBLAS" in code or "cublas" in code or "torch.bmm" in code:
            return f"Delegating matmul to cuBLAS/bmm works (score {score:.4f})"
        if "triton.jit" in code:
            return f"Custom Triton kernel passes correctness (score {score:.4f})"
        return f"Approach works with score {score:.4f}"
    else:
        # Failed — note what went wrong
        if "SIZE MISMATCH" in str(feedback):
            return "Output shape mismatch — check einsum dimensions"
        if "mismatch" in str(feedback).lower():
            return "Numerical mismatch — check precision/dtype handling"
        if "error" in str(feedback).lower():
            return f"Runtime error in generated code"
        return "Failed correctness tests"


def main():
    parser = argparse.ArgumentParser(description="Multi-Strategy TTT with Knowledge Sharing")
    parser.add_argument("--task", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-32B")
    parser.add_argument("--vllm-url", required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--workdir", default="/fsx/xuanj/ttt_diverse_results")
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
    best_code = submission.read_text() if submission.exists() else ""
    best_score = 0.0
    best_feedback = ""
    notebook = []  # shared knowledge across strategies
    strategy_stats = {s["name"]: {"attempts": 0, "wins": 0, "best": 0.0} for s in STRATEGIES}

    log.info("=== Multi-Strategy TTT: %d strategies × %d steps ===", len(STRATEGIES), args.steps)

    for step in range(args.steps):
        log.info("══════ Step %d/%d (best=%.4f) ══════", step + 1, args.steps, best_score)

        step_results = []

        for strategy in STRATEGIES:
            name = strategy["name"]
            log.info("  [%s] Generating...", name)

            try:
                messages = build_prompt(strategy, task_desc, best_code, best_score, best_feedback, notebook)
                response = call_vllm(args.vllm_url, args.model, messages)
                code = extract_code(response)

                if not code or len(code) < 50:
                    log.warning("  [%s] No valid code", name)
                    continue

                score, feedback = run_grader(args.task, code, workdir)
                insight = extract_insight(code, score, feedback, name)
                notebook.append({"strategy": name, "score": score, "insight": insight, "step": step + 1})

                strategy_stats[name]["attempts"] += 1
                if score > strategy_stats[name]["best"]:
                    strategy_stats[name]["best"] = score

                step_results.append((name, code, score, feedback))
                log.info("  [%s] score=%.4f | %s", name, score, insight)

            except Exception as e:
                log.warning("  [%s] Failed: %s", name, e)

        if not step_results:
            continue

        # Pick step winner
        winner_name, winner_code, winner_score, winner_feedback = max(step_results, key=lambda r: r[2])

        if winner_score > best_score:
            best_score = winner_score
            best_code = winner_code
            best_feedback = winner_feedback
            strategy_stats[winner_name]["wins"] += 1
            log.info("  ★ New best: %.4f by [%s]", best_score, winner_name)
        elif winner_score > 0:
            # Even if not best overall, update parent if it's the best this step
            best_code = winner_code
            best_feedback = winner_feedback

        (workdir / f"step_{step+1}_{winner_name}.py").write_text(winner_code)
        (workdir / "best_submission.py").write_text(best_code)

    # Final summary
    log.info("\n=== Final Results ===")
    log.info("Best score: %.4f", best_score)
    log.info("Strategy stats:")
    for name, stats in strategy_stats.items():
        log.info("  [%s] attempts=%d wins=%d best=%.4f",
                 name, stats["attempts"], stats["wins"], stats["best"])
    log.info("Notebook entries: %d", len(notebook))
    log.info("Notebook (last 5):")
    for entry in notebook[-5:]:
        log.info("  [%s] step=%d score=%.4f: %s",
                 entry["strategy"], entry["step"], entry["score"], entry["insight"])


if __name__ == "__main__":
    main()
