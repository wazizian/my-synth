#!/bin/bash
set -euo pipefail
cd /workspace/pallas
trap 'code=$?; echo "$code" > outputs/tensorcore.exit' EXIT
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 PYTHONPATH=/workspace/pallas
unset LD_LIBRARY_PATH
for ((attempt=0; attempt<12000; attempt++)); do
    test -f outputs/repair.exit && break
    sleep 2
done
test -f outputs/repair.exit
cp tensorcore-kernels.py pallas_study/kernels.py
cp tensorcore-benchmark.py pallas_study/benchmark.py
cp tensorcore-meta.py pallas_study/meta_benchmark.py
cp tensorcore-module.py pallas_study/tensorcore.py
mkdir -p versions/v8
cp pallas_study/*.py versions/v8/
for precision in default highest; do
    /workspace/pallas-env/bin/python -m pallas_study.benchmark --batch 128 --steps 126 --analytic-jvp --tensorcore --precision "$precision" --tasks sde_score sde_score_flat sde_pathwise joint --repeats 9 --updates 10 --output "outputs/tensorcore-${precision}-b128.json"
done
/workspace/pallas-env/bin/python -m pallas_study.benchmark --batch 1024 --steps 126 --analytic-jvp --tensorcore --precision highest --tasks sde_score sde_score_flat sde_pathwise joint --repeats 9 --updates 10 --output outputs/tensorcore-highest-b1024.json
/workspace/pallas-env/bin/python -m pallas_study.meta_benchmark --tensorcore --output outputs/meta-tensorcore.json
