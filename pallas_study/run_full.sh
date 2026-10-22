#!/bin/bash
set -euo pipefail
cd /workspace/pallas
trap 'code=$?; echo "$code" > outputs/full.exit' EXIT
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 PYTHONPATH=/workspace/pallas
unset LD_LIBRARY_PATH
for ((attempt=0; attempt<600; attempt++)); do
    test -f outputs/benchmark.exit && break
    sleep 2
done
test "$(cat outputs/benchmark.exit)" = 0
/workspace/pallas-env/bin/python -m pallas_study.validate
for batch in 128 256 1024; do
    /workspace/pallas-env/bin/python -m pallas_study.benchmark --batch "$batch" --steps 126 --repeats 9 --updates 10 --output "outputs/full-b${batch}.json"
done
