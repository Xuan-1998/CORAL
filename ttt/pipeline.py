"""ThetaEvolve-style fixed pipeline for CORAL TTT.

No autonomous agent. Direct: prompt → vLLM → code → CORAL grader → GRPO.

Usage:
  python -m ttt.pipeline \
    --task examples/kernel_engineering/trimul/task.yaml \
    --model Qwen/Qwen3-32B \
    --vllm-url http://<serve_node>:8000/v1 \
    --steps 20
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)


def call_vllm(url, model, messages, max_tokens=4096, temperature=1.0):
    """Call vLLM chat completions API."""
    resp = requests.post(f"{url}/chat/completions", json={
        "model": model, "messages": messages,
        "max_tokens": max_tokens, "temperature": temperature,
    }, timeout=300)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def extract_code(response, lang="python"):
    """Extract code block from LLM response. Handles <think> tags."""
    # Strip thinking tags
    if "</think>" in response:
        response = response[response.index("</think>") + len("</think>"):]

    # Find last python code block (most likely the final answer)
    import re
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    return ""


def run_coral_grader(task_yaml, code, workdir):
    """Run CORAL grader on generated code. Returns (score, feedback)."""
    with open(task_yaml) as f:
        cfg = yaml.safe_load(f)

    grader_args = cfg.get("grader", {}).get("args", {})
    timeout = cfg.get("grader", {}).get("timeout", 600)

    # Write code to submission file
    submission = workdir / "submission.py"
    submission.write_text(code)

    # Import and run grader directly
    eval_dir = Path(task_yaml).parent / "eval"
    private_dir = workdir / ".coral_private"
    private_dir.mkdir(exist_ok=True)
    private_eval = private_dir / "eval"
    if not private_eval.exists():
        import shutil
        shutil.copytree(eval_dir, private_eval)

    # Run grader in subprocess to isolate GPU usage
    grader_script = f"""
import sys, os, json
sys.path.insert(0, '{workdir}')
os.chdir('{workdir}')

# Minimal grader invocation
from pathlib import Path
import yaml, math, shutil, subprocess, tempfile, threading

eval_dir = Path('{private_eval}')
submission_path = '{submission}'

# Load task config
with open(eval_dir / 'task.yml') as f:
    task_config = yaml.safe_load(f)

# Run eval.py
with tempfile.TemporaryDirectory(prefix='coral_kernel_') as tmpdir:
    for f in eval_dir.iterdir():
        if f.name not in ('grader.py',):
            if f.is_file():
                shutil.copy2(f, tmpdir)
    shutil.copy2(submission_path, os.path.join(tmpdir, 'submission.py'))

    # Write test specs
    specs = task_config.get('tests', [])
    with open(os.path.join(tmpdir, 'test.txt'), 'w') as f:
        for spec in specs:
            parts = [f'{{k}}: {{v}}' for k, v in spec.items()]
            f.write('; '.join(parts) + '\\n')

    read_fd, write_fd = os.pipe()
    env = os.environ.copy()
    env['POPCORN_FD'] = str(write_fd)

    proc = subprocess.Popen(
        ['/usr/bin/python3', 'eval.py', 'test', 'test.txt'],
        cwd=tmpdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=env, pass_fds=(write_fd,),
    )
    os.close(write_fd)

    popcorn = []
    def read_popcorn():
        with os.fdopen(read_fd, 'r') as f:
            popcorn.append(f.read())
    t = threading.Thread(target=read_popcorn, daemon=True)
    t.start()

    stdout, stderr = proc.communicate(timeout={timeout})
    t.join(timeout=10)

    results = {{}}
    for line in (popcorn[0] if popcorn else '').strip().splitlines():
        if ':' in line:
            k, _, v = line.partition(':')
            results[k.strip()] = v.strip()

    if results.get('check') != 'pass':
        print(json.dumps({{'score': 0, 'feedback': f'Failed: {{results}}'}}))
    else:
        # Run benchmarks
        bench_specs = task_config.get('benchmarks', [])
        with open(os.path.join(tmpdir, 'leaderboard.txt'), 'w') as f:
            for spec in bench_specs:
                parts = [f'{{k}}: {{v}}' for k, v in spec.items()]
                f.write('; '.join(parts) + '\\n')

        read_fd2, write_fd2 = os.pipe()
        env2 = os.environ.copy()
        env2['POPCORN_FD'] = str(write_fd2)
        proc2 = subprocess.Popen(
            ['/usr/bin/python3', 'eval.py', 'leaderboard', 'leaderboard.txt'],
            cwd=tmpdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env2, pass_fds=(write_fd2,),
        )
        os.close(write_fd2)
        popcorn2 = []
        def read_popcorn2():
            with os.fdopen(read_fd2, 'r') as f:
                popcorn2.append(f.read())
        t2 = threading.Thread(target=read_popcorn2, daemon=True)
        t2.start()
        stdout2, stderr2 = proc2.communicate(timeout={timeout})
        t2.join(timeout=10)

        bench_results = {{}}
        for line in (popcorn2[0] if popcorn2 else '').strip().splitlines():
            if ':' in line:
                k, _, v = line.partition(':')
                bench_results[k.strip()] = v.strip()

        timings = []
        count = int(bench_results.get('benchmark-count', '0'))
        for i in range(count):
            mean_key = f'benchmark.{{i}}.mean'
            if mean_key in bench_results:
                timings.append(float(bench_results[mean_key]))

        if timings:
            from math import log, exp
            geomean_ns = exp(sum(log(v) for v in timings) / len(timings))
            geomean_us = geomean_ns / 1000.0
            score = 1000.0 / geomean_us
            print(json.dumps({{'score': score, 'feedback': f'Runtime: {{geomean_us:.2f}} us, score: {{score:.4f}}'}}))
        else:
            print(json.dumps({{'score': 0, 'feedback': f'No timings: {{bench_results}}'}}))
"""
    result = subprocess.run(
        ["/usr/bin/python3", "-c", grader_script],
        capture_output=True, text=True, timeout=timeout + 60,
        cwd=str(workdir),
    )
    try:
        data = json.loads(result.stdout.strip().split("\n")[-1])
        return data["score"], data["feedback"]
    except Exception:
        return 0, f"Grader error: stdout={result.stdout[-500:]}, stderr={result.stderr[-500:]}"


def build_prompt(task_desc, parent_code, parent_score, parent_feedback, history):
    """Build prompt for code improvement."""
    messages = [{"role": "system", "content": (
        "You are an expert Triton kernel engineer. Generate improved kernel code. "
        "Output ONLY a complete Python file inside ```python ... ``` blocks. "
        "No explanations outside the code block."
    )}]

    user_msg = f"## Task\n{task_desc[:3000]}\n\n"
    user_msg += f"## Current code (score: {parent_score:.4f})\n```python\n{parent_code}\n```\n\n"
    if parent_feedback:
        user_msg += f"## Feedback\n{parent_feedback}\n\n"
    if history:
        user_msg += "## Previous attempts (score, brief):\n"
        for h in history[-5:]:
            user_msg += f"- score={h['score']:.4f}: {h['feedback'][:100]}\n"
    user_msg += "\n## Your task\nImprove the code to get a higher score (lower runtime). Output the complete improved file."

    messages.append({"role": "user", "content": user_msg})
    return messages


def main():
    parser = argparse.ArgumentParser(description="ThetaEvolve-style TTT pipeline")
    parser.add_argument("--task", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-32B")
    parser.add_argument("--vllm-url", required=True, help="vLLM API URL e.g. http://10.0.49.164:8000/v1")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--samples-per-step", type=int, default=4, help="Candidates per step (best-of-N)")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--workdir", default="/tmp/coral_pipeline")
    args = parser.parse_args()

    with open(args.task) as f:
        task_cfg = yaml.safe_load(f)
    task_desc = task_cfg["task"]["description"]

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    # Copy seed files
    seed_dir = Path(args.task).parent / "seed"
    if seed_dir.exists():
        import shutil
        for f in seed_dir.iterdir():
            shutil.copy2(f, workdir)

    # Read initial code
    submission = workdir / "submission.py"
    parent_code = submission.read_text() if submission.exists() else "# empty"
    parent_score = 0.0
    parent_feedback = ""
    history = []
    best_score = 0.0
    best_code = parent_code

    log.info("Starting pipeline: %d steps, %d samples/step", args.steps, args.samples_per_step)

    for step in range(args.steps):
        log.info("══════ Step %d/%d (best=%.4f) ══════", step + 1, args.steps, best_score)

        messages = build_prompt(task_desc, parent_code, parent_score, parent_feedback, history)

        # Sample N candidates
        candidates = []
        for i in range(args.samples_per_step):
            try:
                response = call_vllm(args.vllm_url, args.model, messages,
                                     max_tokens=args.max_tokens, temperature=args.temperature)
                code = extract_code(response)
                if not code or "custom_kernel" not in code:
                    log.warning("  Sample %d: no valid code extracted", i)
                    continue

                # Evaluate
                score, feedback = run_coral_grader(args.task, code, workdir)
                candidates.append({"code": code, "score": score, "feedback": feedback})
                log.info("  Sample %d: score=%.4f %s", i, score, feedback[:80])

            except Exception as e:
                log.warning("  Sample %d failed: %s", i, e)

        if not candidates:
            log.warning("No valid candidates this step")
            continue

        # Pick best candidate
        best_candidate = max(candidates, key=lambda c: c["score"])
        history.append(best_candidate)

        if best_candidate["score"] > best_score:
            best_score = best_candidate["score"]
            best_code = best_candidate["code"]
            log.info("  ★ New best: %.4f", best_score)

        # Update parent for next step (use best from this step)
        if best_candidate["score"] > 0:
            parent_code = best_candidate["code"]
            parent_score = best_candidate["score"]
            parent_feedback = best_candidate["feedback"]

        # Save checkpoint
        (workdir / f"step_{step+1}_best.py").write_text(best_code)
        (workdir / "best_submission.py").write_text(best_code)

    log.info("Done. Best score: %.4f", best_score)
    log.info("Best code saved to %s/best_submission.py", workdir)


if __name__ == "__main__":
    main()
