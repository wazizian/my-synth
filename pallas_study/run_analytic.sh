#!/bin/bash
set -euo pipefail
cd /workspace/pallas
trap 'code=$?; echo "$code" > outputs/analytic.exit' EXIT
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 PYTHONPATH=/workspace/pallas
unset LD_LIBRARY_PATH
for ((attempt=0; attempt<10000; attempt++)); do
    test -f outputs/credit.exit && break
    sleep 2
done
test -f outputs/credit.exit
cp analytic-kernels.py pallas_study/kernels.py
cp analytic-benchmark.py pallas_study/benchmark.py
cp analytic-validate.py pallas_study/validate.py
mkdir -p versions/v4
cp pallas_study/*.py versions/v4/
/workspace/pallas-env/bin/python -m pallas_study.validate --analytic-jvp
for batch in 128 256 1024; do
    /workspace/pallas-env/bin/python -m pallas_study.benchmark --batch "$batch" --steps 126 --analytic-jvp --precision highest --repeats 9 --updates 10 --output "outputs/analytic-high-b${batch}.json"
done
/workspace/pallas-env/bin/python -m pallas_study.benchmark --batch 128 --steps 126 --analytic-jvp --precision highest --dtype float64 --tasks historical --repeats 9 --updates 10 --output outputs/analytic-f64-b128.json
