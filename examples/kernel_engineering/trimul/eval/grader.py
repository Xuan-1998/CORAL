"""Kernel engineering grader — runs eval on remote GPU node via SSH.

Uses shared /fsx filesystem: copies files to a temp dir on /fsx,
then SSHes to GPU node to run eval.py there.
"""
from __future__ import annotations
import logging, math, os, shutil, subprocess, tempfile, threading, json
from pathlib import Path
from typing import Any
import yaml
from coral.grader import TaskGrader
from coral.types import ScoreBundle

logger = logging.getLogger(__name__)

GPU_NODE = os.environ.get("CORAL_GPU_NODE", "p5en-odcr-queue-dy-p5en48xlarge-28")

def _parse_popcorn(output):
    r = {}
    for line in output.strip().splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            r[k.strip()] = v.strip()
    return r

class Grader(TaskGrader):
    def evaluate(self) -> ScoreBundle:
        task_name = self.args.get("task_name", "trimul")
        timeout = self.timeout

        submission_path = os.path.join(self.codebase_path, "submission.py")
        if not os.path.exists(submission_path):
            return self.fail("submission.py not found")

        if "custom_kernel" not in Path(submission_path).read_text():
            return self.fail("submission.py must define custom_kernel")

        task_yml = self.read_eval_path("task.yml")
        with open(task_yml) as f:
            task_config = yaml.safe_load(f)

        eval_dir = Path(self.private_dir) / "eval"

        # Use /fsx temp dir (shared across nodes)
        tmpdir = tempfile.mkdtemp(prefix="coral_kernel_", dir="/fsx/xuanj/tmp")
        try:
            for f in eval_dir.iterdir():
                if f.name != "grader.py" and f.is_file():
                    shutil.copy2(f, tmpdir)
            shutil.copy2(submission_path, os.path.join(tmpdir, "submission.py"))

            # Write test specs
            for mode, key in [("test", "tests"), ("leaderboard", "benchmarks")]:
                specs = task_config.get(key, [])
                with open(os.path.join(tmpdir, f"{mode}.txt"), "w") as f:
                    for spec in specs:
                        f.write("; ".join(f"{k}: {v}" for k, v in spec.items()) + "\n")

            # Run correctness on GPU node via SSH
            logger.info("Running correctness on %s...", GPU_NODE)
            test_result = self._run_remote(tmpdir, "test", timeout)

            if test_result.get("check") != "pass":
                error = self._extract_errors(test_result)
                return self.fail(f"Correctness failed: {error}")

            # Run benchmarks
            logger.info("Running benchmarks on %s...", GPU_NODE)
            bench_result = self._run_remote(tmpdir, "leaderboard", timeout)

            timings = []
            count = int(bench_result.get("benchmark-count", "0"))
            for i in range(count):
                mk = f"benchmark.{i}.mean"
                if mk in bench_result:
                    timings.append(float(bench_result[mk]))

            if not timings:
                return self.fail(f"No timings: {bench_result}")

            geomean_ns = math.exp(sum(math.log(v) for v in timings) / len(timings))
            geomean_us = geomean_ns / 1000.0
            score = 1000.0 / geomean_us
            return self.score(score, f"Runtime: {geomean_us:.2f} us, score: {score:.4f}")

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def _run_remote(self, tmpdir, mode, timeout):
        """Run eval.py on GPU node via SSH, using POPCORN_FD protocol."""
        # Write a wrapper script that handles POPCORN_FD
        wrapper = os.path.join(tmpdir, f"run_{mode}.sh")
        result_file = os.path.join(tmpdir, f"result_{mode}.json")
        with open(wrapper, "w") as f:
            f.write(f"""#!/bin/bash
source /fsx/xuanj/coral-ttt-venv/bin/activate
cd {tmpdir}
python3 -c "
import subprocess, os, threading, json
r,w = os.pipe()
env = os.environ.copy()
env['POPCORN_FD'] = str(w)
p = subprocess.Popen(['/usr/bin/python3','eval.py','{mode}','{mode}.txt'],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, pass_fds=(w,))
os.close(w)
out = []
def rd():
    with os.fdopen(r) as f: out.append(f.read())
t = threading.Thread(target=rd, daemon=True)
t.start()
p.communicate(timeout={timeout})
t.join(5)
res = {{}}
for l in (out[0] if out else '').strip().splitlines():
    if ':' in l: k,_,v = l.partition(':'); res[k.strip()] = v.strip()
with open('{result_file}','w') as f: json.dump(res, f)
"
""")
        os.chmod(wrapper, 0o755)

        try:
            result = subprocess.run(
                ["ssh", "-o", "StrictHostKeyChecking=no", GPU_NODE, f"bash {wrapper}"],
                capture_output=True, text=True, timeout=timeout + 60,
            )
            if os.path.exists(result_file):
                with open(result_file) as f:
                    return json.load(f)
            logger.warning("No result file. stderr: %s", result.stderr[:500])
            return {"check": "fail", "error": result.stderr[:200]}
        except subprocess.TimeoutExpired:
            return {"check": "fail", "error": f"Timeout after {timeout}s"}
        except Exception as e:
            return {"check": "fail", "error": str(e)}

    def _extract_errors(self, results):
        errors = []
        for k, v in results.items():
            if k.endswith(".error"):
                errors.append(v)
        return "; ".join(errors[:3]) if errors else results.get("error", "Unknown")
