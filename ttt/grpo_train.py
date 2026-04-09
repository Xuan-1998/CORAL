"""GRPO training on collected pipeline samples.

Reads (code, score) pairs from pipeline results.
Trains LoRA on the model to increase correctness rate.
Then re-runs pipeline with updated model.

Usage:
  # Step 1: Run pipeline to collect samples (already done)
  # Step 2: Train on collected samples
  python -m ttt.grpo_train \
    --model Qwen/Qwen3-32B \
    --samples-dir /fsx/xuanj/ttt_pipeline_results \
    --task examples/kernel_engineering/trimul/task.yaml \
    --output-dir /fsx/xuanj/ttt_grpo_checkpoint \
    --train-gpu 1 \
    --epochs 3
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)


def collect_samples(samples_dir, task_yaml):
    """Collect (prompt, code, score) triples from pipeline results."""
    samples_dir = Path(samples_dir)
    with open(task_yaml) as f:
        task_desc = yaml.safe_load(f)["task"]["description"]

    # Build the prompt prefix (same as pipeline uses)
    prompt_prefix = (
        "You are an expert Triton kernel engineer. Generate improved kernel code. /no_think "
        "Output ONLY a complete Python file inside ```python ... ``` blocks. "
        "No explanations outside the code block.\n\n"
        f"## Task\n{task_desc[:2000]}\n\n"
        "Generate a complete working Triton kernel implementation.\n"
    )

    samples = []
    for f in sorted(samples_dir.glob("step_*_best.py")):
        code = f.read_text()
        # Determine score from filename or by checking if it's the seed
        # We'll re-evaluate or use a heuristic
        has_triton = "triton" in code.lower() and "@triton.jit" in code
        has_custom_kernel = "custom_kernel" in code
        # Simple heuristic: if it has triton kernels, it was a model-generated attempt
        if has_triton and has_custom_kernel:
            samples.append({"prompt": prompt_prefix, "code": code, "score": 1.0, "source": f.name})
        elif has_custom_kernel:
            samples.append({"prompt": prompt_prefix, "code": code, "score": 0.0, "source": f.name})

    # Also check best_submission.py
    best = samples_dir / "best_submission.py"
    if best.exists():
        code = best.read_text()
        if "triton" in code.lower():
            samples.append({"prompt": prompt_prefix, "code": code, "score": 1.0, "source": "best"})

    log.info("Collected %d samples (%d positive, %d negative)",
             len(samples),
             sum(1 for s in samples if s["score"] > 0),
             sum(1 for s in samples if s["score"] == 0))
    return samples


def grpo_train(model, tokenizer, samples, optimizer, device, epochs=3):
    """GRPO training: upweight good code, downweight bad code."""
    import torch

    if not samples:
        log.warning("No samples to train on")
        return

    # Compute advantages
    scores = [s["score"] for s in samples]
    mean_s = sum(scores) / len(scores)
    std_s = max((sum((s - mean_s) ** 2 for s in scores) / len(scores)) ** 0.5, 1e-8)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n = 0

        for sample in samples:
            adv = (sample["score"] - mean_s) / std_s
            prompt = sample["prompt"]
            code = sample["code"]

            # Tokenize
            full_text = prompt + "```python\n" + code + "\n```"
            enc = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=4096).to(device)
            prompt_len = len(tokenizer(prompt, truncation=True, max_length=2048).input_ids)

            if enc.input_ids.shape[1] <= prompt_len:
                continue

            labels = enc.input_ids.clone()
            labels[0, :prompt_len] = -100  # mask prompt

            out = model(**enc, labels=labels)
            loss = out.loss * adv
            loss.backward()
            total_loss += loss.item()
            n += 1

        if n > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        log.info("Epoch %d/%d: loss=%.4f, n=%d", epoch + 1, epochs, total_loss / max(n, 1), n)


def main():
    parser = argparse.ArgumentParser(description="GRPO training on pipeline samples")
    parser.add_argument("--model", default="Qwen/Qwen3-32B")
    parser.add_argument("--samples-dir", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output-dir", default="/fsx/xuanj/ttt_grpo_checkpoint")
    parser.add_argument("--train-gpu", default="0")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lora-rank", type=int, default=16)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    device = f"cuda:{args.train_gpu}"
    os.makedirs(args.output_dir, exist_ok=True)

    # Collect samples
    samples = collect_samples(args.samples_dir, args.task)
    if not samples:
        log.error("No samples found in %s", args.samples_dir)
        return

    # Load model
    log.info("Loading %s on %s...", args.model, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)

    lora_cfg = LoraConfig(r=args.lora_rank, lora_alpha=32, target_modules=["q_proj", "v_proj"])
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    # Train
    grpo_train(model, tokenizer, samples, optimizer, device, epochs=args.epochs)

    # Save
    log.info("Merging LoRA and saving to %s...", args.output_dir)
    merged = model.merge_and_unload()
    merged.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    log.info("Done. Checkpoint saved.")


if __name__ == "__main__":
    main()
