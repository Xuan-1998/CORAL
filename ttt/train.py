"""CORAL TTT — Test-Time Training with GRPO.

No rllm dependency. Uses vLLM for serving + CORAL for agent orchestration.

Architecture:
  GPU 0: vLLM serves the trainable model (OpenAI-compatible API)
  GPU 1: LoRA training via GRPO on collected trajectories

Flow per step:
  1. CORAL agent (OpenCode) runs, calls vLLM via litellm gateway
  2. Gateway logs all request/response pairs to JSONL
  3. Agent submits eval → CORAL grader scores the attempt
  4. Trainer reads gateway JSONL + attempt scores
  5. Computes reward = score_improvement over parent
  6. GRPO update on LoRA weights
  7. Merge LoRA → restart vLLM with updated weights → repeat
"""

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

POLL_INTERVAL = 10
POLL_TIMEOUT = 1200  # 20 min max per step


# ── vLLM ──────────────────────────────────────────────────────────────

def start_vllm(model_path: str, port: int = 8000, gpu: str = "0") -> subprocess.Popen:
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--port", str(port),
        "--trust-remote-code",
        "--max-model-len", "131072",
        "--dtype", "bfloat16",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "hermes",
    ]
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
                raise RuntimeError("vLLM exited unexpectedly")
            time.sleep(3)
    raise RuntimeError("vLLM startup timeout")


def stop_vllm(proc: subprocess.Popen):
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


# ── CORAL ─────────────────────────────────────────────────────────────

def discover_coral_dir(task_yaml: str) -> Path:
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


def start_coral(task_yaml: str, first: bool) -> subprocess.Popen:
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


def wait_for_evals(coral_dir: Path, n: int, seen: set) -> list[str]:
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


def read_attempts(coral_dir: Path, hashes: list[str]) -> list[dict]:
    out = []
    for h in hashes:
        p = coral_dir / "public" / "attempts" / f"{h}.json"
        if p.exists():
            out.append(json.loads(p.read_text()))
    return sorted(out, key=lambda a: a.get("timestamp", ""))


def read_gateway_traces(coral_dir: Path, offset: int) -> tuple[list[dict], int]:
    log_path = coral_dir / "public" / "gateway" / "requests.jsonl"
    entries, new_offset = [], offset
    if not log_path.exists():
        return entries, new_offset
    with open(log_path) as f:
        for i, line in enumerate(f):
            if i < offset:
                continue
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
                new_offset = i + 1
    return entries, new_offset


def compute_rewards(attempts: list[dict], coral_dir: Path) -> list[float]:
    rewards = []
    for a in attempts:
        score = float(a.get("score") or 0)
        parent_hash = a.get("parent_hash")
        parent_score = 0.0
        if parent_hash:
            pp = coral_dir / "public" / "attempts" / f"{parent_hash}.json"
            if pp.exists():
                parent_score = float(json.loads(pp.read_text()).get("score") or 0)
        rewards.append(score - parent_score)
    return rewards


# ── GRPO ──────────────────────────────────────────────────────────────

def grpo_step(model, tokenizer, traces, rewards, optimizer, device="cuda"):
    """One GRPO gradient step. Returns avg loss."""
    import torch

    if not traces or not rewards:
        return 0.0

    mean_r = sum(rewards) / len(rewards)
    std_r = max((sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5, 1e-8)

    model.train()
    total_loss, n = 0.0, 0

    for trace, reward in zip(traces, rewards):
        adv = (reward - mean_r) / std_r
        req = trace.get("request", {})
        resp = trace.get("response", {})
        messages = req.get("messages", [])
        content = resp.get("content", "") if isinstance(resp, dict) else str(resp)
        if not content or not messages:
            continue

        # Build chat-format input
        parts = [f"<|{m.get('role','user')}|>\n{m.get('content','')}" for m in messages]
        prompt = "\n".join(parts) + "\n<|assistant|>\n"

        enc = tokenizer(prompt + content, return_tensors="pt", truncation=True, max_length=4096).to(device)
        prompt_len = len(tokenizer(prompt, truncation=True, max_length=3072).input_ids)
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

    log.info("GRPO: %d traces, loss=%.4f, mean_reward=%.4f", n, total_loss / max(n, 1), mean_r)
    return total_loss / max(n, 1)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CORAL TTT: Test-Time Training with GRPO")
    parser.add_argument("--task", required=True, help="Path to CORAL task.yaml")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--evals-per-step", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--vllm-gpu", default="0")
    parser.add_argument("--train-gpu", default="1")
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument("--checkpoint-dir", default="./ttt_checkpoints")
    parser.add_argument("--reload-every", type=int, default=5, help="Reload vLLM with new weights every N steps")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    train_device = f"cuda:{args.train_gpu}" if ":" not in args.train_gpu else args.train_gpu

    # 1. Start vLLM
    vllm_proc = start_vllm(args.model, port=args.vllm_port, gpu=args.vllm_gpu)

    # 2. Load model for RL training
    log.info("Loading %s on %s for RL training...", args.model, train_device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True,
    ).to(train_device)
    lora_cfg = LoraConfig(r=args.lora_rank, lora_alpha=32, target_modules=["q_proj", "v_proj"])
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    # 3. Training loop
    seen_hashes: set[str] = set()
    trace_offset = 0
    coral_dir = None
    best_score = 0.0
    all_scores: list[float] = []

    try:
        for step in range(args.steps):
            log.info("══════ Step %d/%d ══════", step + 1, args.steps)

            # Start/resume CORAL
            coral_proc = start_coral(args.task, first=(step == 0))
            if step == 0:
                coral_dir = discover_coral_dir(args.task)

            # Wait for evals
            new_hashes = wait_for_evals(coral_dir, args.evals_per_step, seen_hashes)
            seen_hashes.update(new_hashes)
            stop_coral()
            if coral_proc.poll() is None:
                coral_proc.terminate()
                try:
                    coral_proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    coral_proc.kill()

            # Collect data
            attempts = read_attempts(coral_dir, new_hashes)
            rewards = compute_rewards(attempts, coral_dir)
            traces, trace_offset = read_gateway_traces(coral_dir, trace_offset)

            scores = [float(a.get("score") or 0) for a in attempts]
            all_scores.extend(scores)
            for s in scores:
                if s > best_score:
                    best_score = s
            log.info("Scores: %s | Rewards: %s | Best: %.6f", scores, rewards, best_score)

            # GRPO update
            if traces:
                avg_r = sum(rewards) / len(rewards) if rewards else 0
                grpo_step(model, tokenizer, traces, [avg_r] * len(traces), optimizer, train_device)
            else:
                log.warning("No gateway traces — GRPO skipped (is gateway enabled?)")

            # Checkpoint + reload vLLM
            if (step + 1) % args.reload_every == 0:
                ckpt = Path(args.checkpoint_dir) / f"step_{step+1}"
                log.info("Saving checkpoint to %s and reloading vLLM...", ckpt)
                merged = model.merge_and_unload()
                merged.save_pretrained(ckpt)
                tokenizer.save_pretrained(ckpt)

                # Reload model for training
                model = AutoModelForCausalLM.from_pretrained(
                    str(ckpt), dtype=torch.bfloat16, trust_remote_code=True,
                ).to(train_device)
                model = get_peft_model(model, lora_cfg)
                optimizer = torch.optim.AdamW(
                    [p for p in model.parameters() if p.requires_grad], lr=args.lr
                )

                # Restart vLLM with new weights
                stop_vllm(vllm_proc)
                vllm_proc = start_vllm(str(ckpt), port=args.vllm_port, gpu=args.vllm_gpu)

    except KeyboardInterrupt:
        log.info("Interrupted by user")
    finally:
        stop_coral()
        stop_vllm(vllm_proc)
        final = Path(args.checkpoint_dir) / "final"
        model.save_pretrained(final)
        tokenizer.save_pretrained(final)
        log.info("Done. Best score: %.6f, Total evals: %d", best_score, len(all_scores))
        log.info("All scores: %s", all_scores)


if __name__ == "__main__":
    main()
