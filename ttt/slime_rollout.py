"""CORAL TTT rollout for slime.

Bridges CORAL's agent-based code evolution with slime's RL training.
The model serves via SGLang (slime's rollout engine), CORAL agent calls it,
agent submits code for eval, we collect (prompt, response, reward) as Samples.

Architecture:
  slime training (Megatron) ←→ slime rollout (SGLang) ←→ CORAL agent → grader
                                      ↑
                              This file bridges here

Usage with slime:
  python train.py --custom-rollout-fn ttt.slime_rollout.generate_rollout_coral ...
"""

import asyncio
import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

from slime.rollout.base_types import RolloutFnTrainOutput
from slime.utils.types import Sample


# ── Globals ───────────────────────────────────────────────────────────

_coral_state = {
    "started": False,
    "coral_dir": None,
    "seen_hashes": set(),
    "output_queue": queue.Queue(maxsize=10000),
    "coral_proc": None,
}

TASK_YAML = os.environ.get("CORAL_TASK_YAML", "examples/kernel_engineering/trimul/task_ttt.yaml")
GPU_NODE = os.environ.get("CORAL_GPU_NODE", "")
EVALS_PER_BATCH = int(os.environ.get("CORAL_EVALS_PER_BATCH", "4"))


# ── CORAL management ──────────────────────────────────────────────────

def _discover_coral_dir(task_yaml):
    import yaml
    with open(task_yaml) as f:
        cfg = yaml.safe_load(f)
    results_dir = Path(cfg.get("workspace", {}).get("results_dir", "./results")).resolve()
    name = cfg["task"]["name"]
    slug = "".join(c for c in name.lower().replace(" ", "-") if c.isalnum() or c == "-")
    latest = results_dir / slug / "latest"
    deadline = time.time() + 120
    while time.time() < deadline:
        if latest.exists():
            coral = latest.resolve() / ".coral"
            if coral.is_dir():
                return coral
        time.sleep(2)
    return None


def _start_coral():
    """Start CORAL agent in background."""
    cmd = ["coral", "start", "--config", TASK_YAML, "run.session=local"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    _coral_state["coral_proc"] = proc
    if not _coral_state["coral_dir"]:
        _coral_state["coral_dir"] = _discover_coral_dir(TASK_YAML)
    _coral_state["started"] = True
    return proc


def _resume_coral():
    """Resume CORAL agent."""
    import yaml
    with open(TASK_YAML) as f:
        cfg = yaml.safe_load(f)
    slug = "".join(c for c in cfg["task"]["name"].lower().replace(" ", "-") if c.isalnum() or c == "-")
    cmd = ["coral", "resume", "--task", slug, "run.session=local"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    _coral_state["coral_proc"] = proc
    return proc


def _stop_coral():
    subprocess.run(["coral", "stop"], capture_output=True, timeout=30)
    proc = _coral_state.get("coral_proc")
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


# ── Sample collection ─────────────────────────────────────────────────

def _collect_new_attempts(coral_dir, seen_hashes):
    """Read new attempt JSONs and convert to (prompt, response, reward) tuples."""
    attempts_dir = coral_dir / "public" / "attempts"
    if not attempts_dir.exists():
        return []

    new = []
    for f in attempts_dir.glob("*.json"):
        h = f.stem
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        try:
            d = json.loads(f.read_text())
            score = float(d.get("score") or 0)
            commit = d.get("commit_hash", "")

            # Get the code diff as the "response"
            agent_dirs = list((coral_dir.parent / "agents").glob("agent-*"))
            code = ""
            for ad in agent_dirs:
                try:
                    r = subprocess.run(
                        ["git", "show", f"{commit}:submission.py"],
                        cwd=str(ad), capture_output=True, text=True, timeout=5
                    )
                    if r.stdout:
                        code = r.stdout
                        break
                except Exception:
                    pass

            if code:
                new.append({"code": code, "score": score, "hash": h})
        except Exception:
            pass

    return new


def _attempt_to_sample(attempt, tokenizer, prompt_text):
    """Convert a CORAL attempt to a slime Sample."""
    code = attempt["code"]
    score = attempt["score"]

    # Build prompt + response
    full_text = prompt_text + code
    tokens = tokenizer.encode(full_text)
    prompt_tokens = tokenizer.encode(prompt_text)

    sample = Sample(
        prompt=prompt_text,
        tokens=tokens,
        response=code,
        response_length=len(tokens) - len(prompt_tokens),
        reward=score,
        status=Sample.Status.COMPLETED,
        metadata={"coral_hash": attempt["hash"], "score": score},
    )
    return sample


# ── Main rollout function ─────────────────────────────────────────────

def generate_rollout_coral(args, rollout_id, data_buffer, evaluation=False):
    """slime rollout function that uses CORAL agent for code generation.

    Called by slime's training loop. Returns RolloutFnTrainOutput with
    Samples collected from CORAL agent's eval attempts.
    """
    if evaluation:
        # For eval, just use slime's default SGLang rollout
        from slime.rollout.sglang_rollout import eval_rollout
        from slime.utils.async_utils import run
        eval_output, _ = run(eval_rollout(args, rollout_id))
        return eval_output

    # Prompt for kernel generation
    prompt_text = (
        "You are an expert Triton kernel engineer. "
        "Generate an optimized TriMul kernel.\n"
        "```python\n"
    )

    # Start/resume CORAL
    if not _coral_state["started"]:
        _start_coral()
    else:
        _resume_coral()

    coral_dir = _coral_state["coral_dir"]
    if not coral_dir:
        print("[CORAL rollout] WARNING: no coral_dir found")
        return RolloutFnTrainOutput(samples=[], metrics=None)

    # Wait for N new eval attempts
    deadline = time.time() + 1200  # 20 min timeout
    collected = []
    while len(collected) < EVALS_PER_BATCH and time.time() < deadline:
        new = _collect_new_attempts(coral_dir, _coral_state["seen_hashes"])
        collected.extend(new)
        if len(collected) < EVALS_PER_BATCH:
            time.sleep(10)

    # Stop CORAL (saves sessions for next resume)
    _stop_coral()

    if not collected:
        print("[CORAL rollout] No attempts collected")
        return RolloutFnTrainOutput(samples=[], metrics=None)

    # Convert to slime Samples
    # We need the tokenizer from args
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path or args.model_path, trust_remote_code=True)

    # Group samples: each attempt is one "group" (like one prompt with one response)
    sample_groups = []
    scores = []
    for attempt in collected:
        sample = _attempt_to_sample(attempt, tokenizer, prompt_text)
        sample_groups.append([sample])  # each group has 1 sample
        scores.append(attempt["score"])

    metrics = {
        "rollout/coral_score_mean": sum(scores) / len(scores) if scores else 0,
        "rollout/coral_score_max": max(scores) if scores else 0,
        "rollout/coral_correct_rate": sum(1 for s in scores if s > 0) / len(scores) if scores else 0,
        "rollout/coral_num_attempts": len(collected),
    }

    print(f"[CORAL rollout] Collected {len(collected)} attempts, "
          f"scores={[f'{s:.4f}' for s in scores]}, "
          f"correct={sum(1 for s in scores if s > 0)}/{len(scores)}")

    return RolloutFnTrainOutput(samples=sample_groups, metrics=metrics)
