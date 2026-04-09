"""Cooperative TTT Pipeline: Sequential Refinement + Generator-Critic.

Two agents with different roles alternate on the same code:
  Agent A (Architect): proposes structural changes, new algorithms
  Agent B (Debugger): fixes correctness, optimizes details

Plus a Critic call that reviews code before expensive eval.

Usage:
  python -m ttt.cooperative \
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
    """Run CORAL grader. Returns (score, feedback)."""
    from ttt.pipeline import run_coral_grader
    return run_coral_grader(task_yaml, code, workdir)


# ── Agent Prompts ─────────────────────────────────────────────────────

ARCHITECT_SYSTEM = """You are Agent A — the Architect. Your job is to propose STRUCTURAL improvements to Triton kernel code. /no_think
Focus on:
- Algorithm-level changes (better tiling, fusion strategies, memory access patterns)
- New approaches (different parallelization, mixed precision, cuBLAS delegation)
- Bold changes that might break things but could lead to big speedups

You will be paired with a Debugger (Agent B) who fixes your code.
Output ONLY a complete Python file in ```python ... ``` blocks."""

DEBUGGER_SYSTEM = """You are Agent B — the Debugger. Your job is to FIX correctness issues in kernel code. /no_think
Focus on:
- Making the code pass all correctness tests (rtol=2e-2, atol=2e-2)
- Fixing shape mismatches, dtype issues, boundary conditions
- Preserving the Architect's structural choices while fixing bugs

Do NOT rewrite the algorithm — fix the bugs in what the Architect gave you.
Output ONLY a complete Python file in ```python ... ``` blocks."""

CRITIC_SYSTEM = """You are the Critic. Review this Triton kernel code and predict issues. /no_think
Output a brief analysis (max 200 words):
1. Will it compile? (yes/no + why)
2. Will it produce correct output? (yes/no + likely issues)
3. One specific fix suggestion.
Do NOT output code."""


def build_architect_prompt(task_desc, parent_code, parent_score, feedback, history):
    msg = f"## Task\n{task_desc[:2000]}\n\n"
    msg += f"## Current code (score: {parent_score:.4f})\n```python\n{parent_code[:3000]}\n```\n\n"
    if feedback:
        msg += f"## Last eval feedback\n{feedback[:500]}\n\n"
    if history:
        msg += "## Score history:\n"
        for h in history[-5:]:
            msg += f"- Step {h['step']}: {h['agent']} → score={h['score']:.4f}\n"
    msg += "\nPropose a STRUCTURAL improvement. Be bold — your Debugger partner will fix any bugs."
    return [{"role": "system", "content": ARCHITECT_SYSTEM}, {"role": "user", "content": msg}]


def build_debugger_prompt(task_desc, architect_code, architect_feedback, parent_code, parent_score):
    msg = f"## Task\n{task_desc[:1500]}\n\n"
    msg += f"## Architect's new code (NOT YET TESTED)\n```python\n{architect_code[:4000]}\n```\n\n"
    if architect_feedback:
        msg += f"## Critic's review of Architect's code\n{architect_feedback}\n\n"
    msg += f"## Previous working code (score: {parent_score:.4f})\n```python\n{parent_code[:2000]}\n```\n\n"
    msg += "Fix any bugs in the Architect's code. Make it pass correctness tests. Keep the structural changes."
    return [{"role": "system", "content": DEBUGGER_SYSTEM}, {"role": "user", "content": msg}]


def build_critic_prompt(code):
    msg = f"Review this Triton kernel code:\n```python\n{code[:4000]}\n```\n\nWill it work correctly?"
    return [{"role": "system", "content": CRITIC_SYSTEM}, {"role": "user", "content": msg}]


# ── Main Loop ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cooperative TTT: Architect + Debugger + Critic")
    parser.add_argument("--task", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-32B")
    parser.add_argument("--vllm-url", required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--workdir", default="/fsx/xuanj/ttt_coop_results")
    args = parser.parse_args()

    with open(args.task) as f:
        task_cfg = yaml.safe_load(f)
    task_desc = task_cfg["task"]["description"]

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    # Copy seed
    seed_dir = Path(args.task).parent / "seed"
    if seed_dir.exists():
        import shutil
        for f in seed_dir.iterdir():
            if f.is_file():
                shutil.copy2(f, workdir)

    submission = workdir / "submission.py"
    parent_code = submission.read_text() if submission.exists() else ""
    parent_score = 0.0
    parent_feedback = ""
    best_score = 0.0
    best_code = parent_code
    history = []

    log.info("=== Cooperative TTT: Architect + Debugger + Critic ===")

    for step in range(args.steps):
        log.info("══════ Step %d/%d (best=%.4f) ══════", step + 1, args.steps, best_score)

        # ── Phase 1: Architect proposes ──
        log.info("  [Architect] Proposing structural change...")
        arch_msgs = build_architect_prompt(task_desc, parent_code, parent_score, parent_feedback, history)
        arch_response = call_vllm(args.vllm_url, args.model, arch_msgs)
        arch_code = extract_code(arch_response)

        if not arch_code or len(arch_code) < 50:
            log.warning("  [Architect] No valid code, skipping step")
            continue

        # ── Phase 2: Critic reviews ──
        log.info("  [Critic] Reviewing Architect's code...")
        critic_msgs = build_critic_prompt(arch_code)
        critic_review = call_vllm(args.vllm_url, args.model, critic_msgs, max_tokens=512, temperature=0.3)
        if "</think>" in critic_review:
            critic_review = critic_review[critic_review.index("</think>") + 8:]
        log.info("  [Critic] %s", critic_review[:150])

        # ── Phase 3: Debugger fixes ──
        log.info("  [Debugger] Fixing Architect's code...")
        debug_msgs = build_debugger_prompt(task_desc, arch_code, critic_review, parent_code, parent_score)
        debug_response = call_vllm(args.vllm_url, args.model, debug_msgs)
        debug_code = extract_code(debug_response)

        if not debug_code or len(debug_code) < 50:
            log.warning("  [Debugger] No valid code, trying Architect's code directly")
            debug_code = arch_code

        # ── Phase 4: Evaluate both ──
        results = []

        # Eval Architect's raw code
        log.info("  [Eval] Testing Architect's code...")
        a_score, a_feedback = run_grader(args.task, arch_code, workdir)
        results.append(("Architect", arch_code, a_score, a_feedback))
        log.info("  [Architect] score=%.4f %s", a_score, a_feedback[:100])

        # Eval Debugger's fixed code
        log.info("  [Eval] Testing Debugger's code...")
        d_score, d_feedback = run_grader(args.task, debug_code, workdir)
        results.append(("Debugger", debug_code, d_score, d_feedback))
        log.info("  [Debugger] score=%.4f %s", d_score, d_feedback[:100])

        # Pick winner
        winner = max(results, key=lambda r: r[2])
        agent_name, winner_code, winner_score, winner_feedback = winner

        history.append({"step": step + 1, "agent": agent_name, "score": winner_score,
                        "arch_score": a_score, "debug_score": d_score})

        if winner_score > 0:
            parent_code = winner_code
            parent_score = winner_score
            parent_feedback = winner_feedback

        if winner_score > best_score:
            best_score = winner_score
            best_code = winner_code
            log.info("  ★ New best: %.4f (by %s)", best_score, agent_name)

        (workdir / f"step_{step+1}_{agent_name}.py").write_text(winner_code)
        (workdir / "best_submission.py").write_text(best_code)

        # Log cooperation stats
        log.info("  Arch=%.4f Debug=%.4f Winner=%s | Critic helped: %s",
                 a_score, d_score, agent_name,
                 "yes" if d_score > a_score else "no")

    # Summary
    log.info("\n=== Final Results ===")
    log.info("Best score: %.4f", best_score)
    arch_wins = sum(1 for h in history if h["agent"] == "Architect")
    debug_wins = sum(1 for h in history if h["agent"] == "Debugger")
    log.info("Architect wins: %d, Debugger wins: %d", arch_wins, debug_wins)
    coop_helped = sum(1 for h in history if h["debug_score"] > h["arch_score"])
    log.info("Debugger improved Architect's code: %d/%d steps", coop_helped, len(history))
    log.info("Best code: %s/best_submission.py", workdir)


if __name__ == "__main__":
    main()
