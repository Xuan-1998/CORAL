"""CORAL TTT — Test-Time Training with GRPO.

Trajectory-level RL: uses CORAL attempt diffs as training signal.
No gateway traces needed — reads code diffs from git commits.

Each step:
  1. CORAL agent runs, produces eval attempts
  2. Read the code diff for each attempt (what the agent changed)
  3. Reward = score improvement over parent
  4. Fine-tune model: prompt = task description + parent code, completion = diff
  5. Weight by advantage (GRPO)
  6. Periodically merge LoRA + restart vLLM
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

POLL_INTERVAL = 10
POLL_TIMEOUT = 1200


# ── vLLM ──────────────────────────────────────────────────────────────

def start_vllm(model_path, port=8000, gpu="0"):
    cmd = ["python", "-m", "vllm.entrypoints.openai.api_server",
           "--model", model_path, "--port", str(port), "--trust-remote-code",
           "--max-model-len", "131072", "--dtype", "bfloat16",
           "--enable-auto-tool-choice", "--tool-call-parser", "hermes"]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    log.info("Starting vLLM on GPU %s: %s", gpu, model_path)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    deadline = time.time() + 240
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://localhost:{port}/health")
            log.info("vLLM ready on port %d", port)
            return proc
        except Exception:
            if proc.poll() is not None:
                raise RuntimeError("vLLM exited")
            time.sleep(3)
    raise RuntimeError("vLLM timeout")


def stop_vllm(proc):
    if proc and proc.poll() is None:
        proc.terminate()
        try: proc.wait(timeout=15)
        except: proc.kill(); proc.wait()


# ── CORAL ─────────────────────────────────────────────────────────────

def discover_coral_dir(task_yaml):
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
    raise RuntimeError(f"No .coral dir under {results_dir / slug}")


def start_coral(task_yaml, first):
    if first:
        cmd = ["coral", "start", "--config", task_yaml, "run.session=local"]
    else:
        with open(task_yaml) as f:
            cfg = yaml.safe_load(f)
        slug = "".join(c for c in cfg["task"]["name"].lower().replace(" ", "-") if c.isalnum() or c == "-")
        cmd = ["coral", "resume", "--task", slug, "run.session=local"]
    log.info("CORAL: %s", " ".join(cmd))
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def stop_coral():
    subprocess.run(["coral", "stop"], capture_output=True, timeout=30)


def wait_for_evals(coral_dir, n, seen):
    attempts_dir = coral_dir / "public" / "attempts"
    deadline = time.time() + POLL_TIMEOUT
    while time.time() < deadline:
        if attempts_dir.exists():
            new = {f.stem for f in attempts_dir.glob("*.json")} - seen
            if len(new) >= n:
                return sorted(new)
        time.sleep(POLL_INTERVAL)
    if attempts_dir.exists():
        return sorted({f.stem for f in attempts_dir.glob("*.json")} - seen)
    return []


def read_attempts(coral_dir, hashes):
    out = []
    for h in hashes:
        p = coral_dir / "public" / "attempts" / f"{h}.json"
        if p.exists():
            out.append(json.loads(p.read_text()))
    return sorted(out, key=lambda a: a.get("timestamp", ""))


def get_attempt_diff(coral_dir, attempt):
    """Get the git diff for an attempt (what code the agent changed)."""
    commit = attempt.get("commit_hash", "")
    if not commit:
        return ""
    # Find the agent worktree
    run_dir = coral_dir.parent
    agent_dirs = list((run_dir / "agents").glob("agent-*"))
    for agent_dir in agent_dirs:
        try:
            result = subprocess.run(
                ["git", "diff", f"{commit}~1", commit, "--", "*.py"],
                cwd=str(agent_dir), capture_output=True, text=True, timeout=10
            )
            if result.stdout.strip():
                return result.stdout.strip()
        except Exception:
            pass
    return ""


# ── GRPO on attempt diffs ─────────────────────────────────────────────

def grpo_step(model, tokenizer, attempts, coral_dir, optimizer, device, task_desc):
    """GRPO using attempt diffs as training data.

    prompt = task description + "improve this code"
    completion = the diff the agent applied
    advantage = normalized score improvement
    """
    import torch

    # Compute rewards
    rewards = []
    diffs = []
    for a in attempts:
        score = float(a.get("score") or 0)
        parent_hash = a.get("parent_hash")
        parent_score = 0.0
        if parent_hash:
            pp = coral_dir / "public" / "attempts" / f"{parent_hash}.json"
            if pp.exists():
                parent_score = float(json.loads(pp.read_text()).get("score") or 0)
        rewards.append(score - parent_score)
        diffs.append(get_attempt_diff(coral_dir, a))

    # Filter out empty diffs
    valid = [(r, d) for r, d in zip(rewards, diffs) if d]
    if not valid:
        log.warning("No valid diffs for GRPO")
        return 0.0

    rewards_v = [r for r, _ in valid]
    diffs_v = [d for _, d in valid]

    mean_r = sum(rewards_v) / len(rewards_v)
    std_r = max((sum((r - mean_r) ** 2 for r in rewards_v) / len(rewards_v)) ** 0.5, 1e-8)

    model.train()
    total_loss, n = 0.0, 0

    for reward, diff in zip(rewards_v, diffs_v):
        adv = (reward - mean_r) / std_r

        # Prompt: task + instruction to improve
        prompt = f"Task: {task_desc[:500]}\n\nImprove the code. Output a diff:\n"
        completion = diff[:2000]  # truncate long diffs

        enc = tokenizer(prompt + completion, return_tensors="pt", truncation=True, max_length=4096).to(device)
        prompt_len = len(tokenizer(prompt, truncation=True, max_length=2048).input_ids)
        if enc.input_ids.shape[1] <= prompt_len:
            continue

        labels = enc.input_ids.clone()
        labels[0, :prompt_len] = -100

        out = model(**enc, labels=labels)
        loss = out.loss * adv
        loss.backward()
        total_loss += loss.item()
        n += 1

    if n > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

    log.info("GRPO: %d diffs, loss=%.4f, mean_reward=%.4f, rewards=%s", n, total_loss / max(n, 1), mean_r, rewards_v)
    return total_loss / max(n, 1)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CORAL TTT")
    parser.add_argument("--task", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--evals-per-step", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--vllm-gpu", default="0")
    parser.add_argument("--train-gpu", default="1")
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument("--checkpoint-dir", default="./ttt_checkpoints")
    parser.add_argument("--reload-every", type=int, default=5)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    train_device = f"cuda:{args.train_gpu}"

    # Read task description for prompts
    with open(args.task) as f:
        task_cfg = yaml.safe_load(f)
    task_desc = task_cfg["task"].get("description", "")

    # 1. Start vLLM
    vllm_proc = start_vllm(args.model, port=args.vllm_port, gpu=args.vllm_gpu)

    # 2. Load model for training
    log.info("Loading %s on %s...", args.model, train_device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True
    ).to(train_device)
    lora_cfg = LoraConfig(r=args.lora_rank, lora_alpha=32, target_modules=["q_proj", "v_proj"])
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    # 3. Training loop
    seen_hashes = set()
    coral_dir = None
    best_score = 0.0
    all_scores = []

    try:
        for step in range(args.steps):
            log.info("══════ Step %d/%d ══════", step + 1, args.steps)

            coral_proc = start_coral(args.task, first=(step == 0))
            if step == 0:
                coral_dir = discover_coral_dir(args.task)

            new_hashes = wait_for_evals(coral_dir, args.evals_per_step, seen_hashes)
            seen_hashes.update(new_hashes)
            stop_coral()
            if coral_proc.poll() is None:
                coral_proc.terminate()
                try: coral_proc.wait(timeout=10)
                except: coral_proc.kill()

            attempts = read_attempts(coral_dir, new_hashes)
            scores = [float(a.get("score") or 0) for a in attempts]
            all_scores.extend(scores)
            for s in scores:
                if s > best_score:
                    best_score = s
            log.info("Scores: %s | Best: %.6f | Total evals: %d", scores, best_score, len(all_scores))

            # GRPO on diffs
            if attempts:
                grpo_step(model, tokenizer, attempts, coral_dir, optimizer, train_device, task_desc)

            # Checkpoint + reload
            if (step + 1) % args.reload_every == 0:
                ckpt = Path(args.checkpoint_dir) / f"step_{step+1}"
                log.info("Checkpoint → %s, reloading vLLM...", ckpt)
                merged = model.merge_and_unload()
                merged.save_pretrained(ckpt)
                tokenizer.save_pretrained(ckpt)
                model = AutoModelForCausalLM.from_pretrained(
                    str(ckpt), dtype=torch.bfloat16, trust_remote_code=True
                ).to(train_device)
                model = get_peft_model(model, lora_cfg)
                optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
                stop_vllm(vllm_proc)
                vllm_proc = start_vllm(str(ckpt), port=args.vllm_port, gpu=args.vllm_gpu)

    except KeyboardInterrupt:
        log.info("Interrupted")
    finally:
        stop_coral()
        stop_vllm(vllm_proc)
        final = Path(args.checkpoint_dir) / "final"
        model.save_pretrained(final)
        tokenizer.save_pretrained(final)
        log.info("Done. Best=%.6f, Evals=%d, Scores=%s", best_score, len(all_scores), all_scores)


if __name__ == "__main__":
    main()
