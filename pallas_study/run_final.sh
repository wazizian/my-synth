#!/bin/bash
set -euo pipefail
cd /workspace/pallas
trap 'code=$?; echo "$code" > outputs/final.exit' EXIT
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 PYTHONPATH=/workspace/pallas
unset LD_LIBRARY_PATH
for ((attempt=0; attempt<12000; attempt++)); do
    test -f outputs/persistent-retry.exit && break
    sleep 2
done
test -f outputs/persistent-retry.exit
cp final-kernels.py pallas_study/kernels.py
cp final-credit-benchmark.py pallas_study/credit_benchmark.py
mkdir -p versions/v6
cp pallas_study/*.py versions/v6/
/workspace/pallas-env/bin/python -m pallas_study.validate --analytic-jvp
for batch in 128 1024; do
    /workspace/pallas-env/bin/python -m pallas_study.benchmark --batch "$batch" --steps 126 --analytic-jvp --precision highest --repeats 9 --updates 10 --output "outputs/final-high-b${batch}.json"
done
/workspace/pallas-env/bin/python -m pallas_study.meta_benchmark
for task in hedger sde; do
    /workspace/pallas-env/bin/python -m pallas_study.credit_benchmark --task "$task" --analytic-only --output "outputs/credit-${task}-analytic.json"
done
/workspace/pallas-env/bin/python -m pallas_study.benchmark --batch 128 --steps 126 --analytic-jvp --precision highest --dtype float64 --tasks historical --repeats 9 --updates 10 --output outputs/final-f64-b128.json
