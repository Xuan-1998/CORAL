#!/bin/bash
# Collect high-quality kernel samples using kiro-cli (Claude) on head node
# Eval runs on GPU node via srun
# Usage: bash ttt/collect_claude.sh <SLURM_JOB_ID>
set -euo pipefail

JOBID=${1:?Usage: $0 <SLURM_JOB_ID>}
GPU_NODE=${2:-p5en-odcr-queue-dy-p5en48xlarge-28}
export PATH=/opt/slurm/bin:$HOME/.local/bin:$HOME/.npm-global/bin:$PATH

WORKDIR=/fsx/xuanj/claude_kernel_samples
TASK_DESC="Implement an optimized Triton kernel for TriMul (Triangle Multiplicative Update).
The function custom_kernel(data) takes (input[B,N,N,C], mask[B,N,N], weights, config) and returns output[B,N,N,C].
Key: fuse LayerNorm+projections+gating, use FP16 matmul for TensorCores, minimize memory traffic.
Score = 1000/geometric_mean(runtime_us). Current best human: ~1371µs (score ~0.73)."

SEED_FILE=/fsx/xuanj/CORAL-xuan/examples/kernel_engineering/trimul/seed/submission.py
EVAL_DIR=/fsx/xuanj/CORAL-xuan/examples/kernel_engineering/trimul/eval

mkdir -p $WORKDIR/samples $WORKDIR/logs

echo "=== Claude Kernel Collection ==="
echo "GPU node: $GPU_NODE, Job: $JOBID"

for i in $(seq 1 20); do
    echo "[$(date)] === Sample $i/20 ==="

    # Generate kernel with kiro-cli on head node
    PROMPT="You are an expert Triton kernel engineer. Write an optimized TriMul kernel.

$TASK_DESC

Here is the reference implementation:
\`\`\`python
$(head -80 $SEED_FILE)
\`\`\`

Write a COMPLETE optimized submission.py with custom_kernel(data) function.
Use Triton @triton.jit kernels for fused operations and FP16 matmul.
Output ONLY the complete Python file in \`\`\`python\`\`\` blocks."

    echo "$PROMPT" | timeout 120 kiro-cli chat --no-interactive -a --model claude-opus-4.6 2>/dev/null > $WORKDIR/logs/response_$i.txt || true

    # Extract code
    python3 -c "
import re, sys
with open('$WORKDIR/logs/response_$i.txt') as f:
    content = f.read()
blocks = re.findall(r'\`\`\`(?:python)?\s*\n(.*?)\`\`\`', content, re.DOTALL)
if blocks:
    code = blocks[-1].strip()
    if 'custom_kernel' in code and len(code) > 100:
        with open('$WORKDIR/samples/sample_$i.py', 'w') as f:
            f.write(code)
        print(f'Extracted {len(code)} chars')
    else:
        print('No valid custom_kernel found')
else:
    print('No code blocks found')
" || echo "extraction failed"

    if [ ! -f "$WORKDIR/samples/sample_$i.py" ]; then
        echo "  No valid code, skipping eval"
        continue
    fi

    # Eval on GPU node
    echo "  Evaluating on $GPU_NODE..."
    SCORE=$(srun --jobid=$JOBID --nodelist=$GPU_NODE --ntasks=1 --gpus=1 --overlap bash -c "
        source /fsx/xuanj/coral-ttt-venv/bin/activate
        cd /tmp
        cp -r $EVAL_DIR/* .
        cp $WORKDIR/samples/sample_$i.py submission.py
        /usr/bin/python3 -c \"
import subprocess, os, threading, json, math
os.environ['POPCORN_FD'] = '3'
# Test correctness first
r,w = os.pipe()
env = os.environ.copy()
env['POPCORN_FD'] = str(w)
p = subprocess.Popen(['/usr/bin/python3','eval.py','test','test.txt'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, pass_fds=(w,))
os.close(w)
out = []
def rd():
    with os.fdopen(r) as f: out.append(f.read())
t = threading.Thread(target=rd, daemon=True)
t.start()
p.communicate(timeout=300)
t.join(5)
res = {}
for l in (out[0] if out else '').strip().splitlines():
    if ':' in l: k,_,v = l.partition(':'); res[k.strip()] = v.strip()
if res.get('check') != 'pass':
    print('0')
else:
    # Benchmark
    r2,w2 = os.pipe()
    env2 = os.environ.copy()
    env2['POPCORN_FD'] = str(w2)
    p2 = subprocess.Popen(['/usr/bin/python3','eval.py','leaderboard','leaderboard.txt'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env2, pass_fds=(w2,))
    os.close(w2)
    out2 = []
    def rd2():
        with os.fdopen(r2) as f: out2.append(f.read())
    t2 = threading.Thread(target=rd2, daemon=True)
    t2.start()
    p2.communicate(timeout=600)
    t2.join(5)
    br = {}
    for l in (out2[0] if out2 else '').strip().splitlines():
        if ':' in l: k,_,v = l.partition(':'); br[k.strip()] = v.strip()
    timings = []
    for j in range(int(br.get('benchmark-count','0'))):
        mk = f'benchmark.{j}.mean'
        if mk in br: timings.append(float(br[mk]))
    if timings:
        gm = math.exp(sum(math.log(v) for v in timings)/len(timings))
        print(f'{1000.0/(gm/1000.0):.4f}')
    else:
        print('0')
\" 2>/dev/null
    " 2>/dev/null || echo "0")

    echo "  Score: $SCORE"
    echo "$i $SCORE" >> $WORKDIR/scores.txt

    if [ "$(echo "$SCORE > 0" | bc -l 2>/dev/null || echo 0)" = "1" ]; then
        cp $WORKDIR/samples/sample_$i.py $WORKDIR/samples/correct_$i.py
        echo "  ✓ Correct kernel!"
    fi
done

echo "=== Collection Done ==="
echo "Scores:"
cat $WORKDIR/scores.txt 2>/dev/null
echo "Correct samples:"
ls $WORKDIR/samples/correct_*.py 2>/dev/null | wc -l
