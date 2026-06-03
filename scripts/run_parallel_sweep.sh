#!/bin/bash
# Parallel orchestration of baseline + lambda sweep.
# Runs the baseline and 9 PINN lambda jobs with a concurrency limit,
# packing the GPU (each job is GPU-resident). After all jobs finish,
# aggregates per-lambda metrics into lambda_sweep.json.
#
# Usage:  bash scripts/run_parallel_sweep.sh
set -u
cd "$(dirname "$0")/.."

MAX_CONCURRENT=3
LOGDIR=/tmp/pinn_run
mkdir -p "$LOGDIR"
LAMBDAS=(0.0 0.01 0.05 0.08 0.1 0.15 0.2 0.5 1.0)

echo "[$(date +%H:%M:%S)] Starting parallel sweep — max ${MAX_CONCURRENT} concurrent"

# Baseline (writes outputs/baseline/)
docker compose run --rm dev python scripts/03_train_baseline.py \
    > "$LOGDIR/baseline.log" 2>&1 &
echo "[$(date +%H:%M:%S)] launched baseline"

# PINN lambdas (each writes outputs/pinn/lambda_X/)
for lam in "${LAMBDAS[@]}"; do
    while [ "$(jobs -r | wc -l)" -ge "$MAX_CONCURRENT" ]; do
        wait -n
    done
    docker compose run --rm dev python scripts/04_train_pinn.py --lambda_phys "$lam" \
        > "$LOGDIR/lambda_$lam.log" 2>&1 &
    echo "[$(date +%H:%M:%S)] launched lambda=$lam"
done

wait
echo "[$(date +%H:%M:%S)] all training jobs finished"

# Aggregate per-lambda metrics into lambda_sweep.json
docker compose run --rm dev python - <<'PY'
import json
from pathlib import Path

lambdas = [0.0, 0.01, 0.05, 0.08, 0.1, 0.15, 0.2, 0.5, 1.0]
summary = {}
for lam in lambdas:
    path = Path("outputs/pinn") / f"lambda_{lam}" / "metrics.json"
    if not path.exists():
        print(f"  WARN missing {path}")
        continue
    data = json.loads(path.read_text())
    summary[str(lam)] = {"lambda_phys": lam, "aggregate": data["aggregate"]}

out = Path("outputs/pinn/lambda_sweep.json")
out.write_text(json.dumps(summary, indent=2))
print(f"Aggregated {len(summary)} lambdas -> {out}")
PY

echo "[$(date +%H:%M:%S)] DONE — lambda_sweep.json written"
