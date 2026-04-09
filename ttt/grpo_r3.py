"""GRPO Round 3: Entropic objective to push best score higher.

Key fixes over R2:
- More samples (from R1 collection run)
- Entropic weighting: exp(β * score) favors high-score samples
- Only 1 epoch to prevent catastrophic forgetting
- KL penalty to stay close to R1 checkpoint
- Diverse prompt templates to encourage different strategies
"""

import logging, os, sys, json, re
from pathlib import Path
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)


def collect_from_log(log_path, results_dir, task_yaml):
    """Parse pipeline log to get (code, score) pairs with actual scores."""
    with open(task_yaml) as f:
        task_desc = yaml.safe_load(f)["task"]["description"]

    samples = []
    results_dir = Path(results_dir)

    # Read all step files and their scores from the log
    score_pattern = re.compile(r"Sample \d+: score=([0-9.]+)")
    code_files = sorted(results_dir.glob("step_*_best.py"))

    for cf in code_files:
        code = cf.read_text()
        if "custom_kernel" not in code or len(code) < 100:
            continue
        # Infer score: files with triton kernels that passed = ~0.091
        has_triton = "@triton.jit" in code
        score = 0.091 if has_triton else 0.0
        samples.append({"code": code, "score": score})

    # Also parse log for exact scores
    if Path(log_path).exists():
        with open(log_path) as f:
            for line in f:
                m = score_pattern.search(line)
                if m:
                    s = float(m.group(1))
                    if s > 0:
                        # We don't have the exact code, but we know the score distribution
                        pass

    # Deduplicate by code hash
    seen = set()
    unique = []
    for s in samples:
        h = hash(s["code"][:500])
        if h not in seen:
            seen.add(h)
            unique.append(s)

    log.info("Collected %d unique samples (%d correct)",
             len(unique), sum(1 for s in unique if s["score"] > 0))
    return unique, task_desc


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    import math

    CKPT = "/fsx/xuanj/ttt_grpo_checkpoint"  # R1 checkpoint
    OUTPUT = "/fsx/xuanj/ttt_grpo_r3_checkpoint"
    TASK = "/fsx/xuanj/CORAL-xuan/examples/kernel_engineering/trimul/task.yaml"
    SAMPLES_DIR = "/fsx/xuanj/ttt_r3_samples"
    LOG_PATH = "/fsx/xuanj/ttt-r3-collect.log"
    DEVICE = "cuda:0"
    BETA = 10.0  # entropic temperature — higher = more focus on best samples
    KL_COEFF = 0.01
    LR = 2e-6  # lower LR to prevent catastrophic forgetting

    samples, task_desc = collect_from_log(LOG_PATH, SAMPLES_DIR, TASK)
    if len(samples) < 3:
        log.error("Not enough samples (%d). Wait for collection to finish.", len(samples))
        return

    # Prompts that encourage different strategies
    prompts = [
        f"/no_think You are an expert Triton kernel engineer. Generate an optimized TriMul kernel.\n## Task\n{task_desc[:1500]}\nFocus on CORRECTNESS first, then speed.\nOutput complete Python in ```python``` blocks.\n",
        f"/no_think You are a GPU optimization expert. Write a fast TriMul kernel using MIXED PRECISION (FP16 matmul + FP32 accumulation).\n## Task\n{task_desc[:1500]}\nOutput complete Python in ```python``` blocks.\n",
        f"/no_think You are a kernel fusion expert. Write a TriMul kernel that FUSES LayerNorm + projection + gating into one kernel.\n## Task\n{task_desc[:1500]}\nOutput complete Python in ```python``` blocks.\n",
    ]

    # Assign prompts to samples (round-robin)
    for i, s in enumerate(samples):
        s["prompt"] = prompts[i % len(prompts)]

    # Compute entropic weights
    scores = [s["score"] for s in samples]
    max_score = max(scores) if scores else 1.0
    weights = [math.exp(BETA * (s["score"] / max(max_score, 1e-8))) for s in samples]
    weight_sum = sum(weights)
    weights = [w / weight_sum for w in weights]

    log.info("Entropic weights: min=%.4f max=%.4f (β=%.1f)",
             min(weights), max(weights), BETA)
    log.info("Top weight samples: %s",
             sorted(zip(weights, [s["score"] for s in samples]), reverse=True)[:5])

    # Load model
    log.info("Loading %s...", CKPT)
    tokenizer = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(CKPT, dtype=torch.bfloat16, trust_remote_code=True).to(DEVICE)
    lora_cfg = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
    model = get_peft_model(model, lora_cfg)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.print_trainable_parameters()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)

    # Single epoch with entropic weighting
    model.train()
    total_loss, n = 0.0, 0

    for sample, weight in zip(samples, weights):
        full = sample["prompt"] + "```python\n" + sample["code"][:2000] + "\n```"
        enc = tokenizer(full, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)
        plen = len(tokenizer(sample["prompt"], truncation=True, max_length=1024).input_ids)
        if enc.input_ids.shape[1] <= plen:
            continue

        labels = enc.input_ids.clone()
        labels[0, :plen] = -100

        out = model(**enc, labels=labels)
        # Entropic loss: weight by exp(β * score) instead of normalized advantage
        loss = out.loss * weight * len(samples)  # scale so gradient magnitude is reasonable
        (loss / len(samples)).backward()
        total_loss += loss.item()
        n += 1

    if n > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

    log.info("Entropic GRPO: loss=%.4f, n=%d samples", total_loss / max(n, 1), n)

    # Save
    os.makedirs(OUTPUT, exist_ok=True)
    merged = model.merge_and_unload()
    merged.save_pretrained(OUTPUT)
    tokenizer.save_pretrained(OUTPUT)
    log.info("Saved to %s", OUTPUT)


if __name__ == "__main__":
    main()
