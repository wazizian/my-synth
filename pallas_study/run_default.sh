#!/bin/bash
set -euo pipefail
cd /workspace/pallas
trap 'code=$?; echo "$code" > outputs/default.exit' EXIT
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 PYTHONPATH=/workspace/pallas
unset LD_LIBRARY_PATH
for ((attempt=0; attempt<12000; attempt++)); do
    test -f outputs/persistent-diagnostic.exit && break
    sleep 2
done
test -f outputs/persistent-diagnostic.exit
for batch in 128 1024; do
    /workspace/pallas-env/bin/python -m pallas_study.benchmark --batch "$batch" --steps 126 --analytic-jvp --precision default --repeats 9 --updates 10 --output "outputs/final-default-b${batch}.json"
done
/workspace/pallas-env/bin/python -m pallas_study.benchmark --batch 128 --steps 126 --saved-backward --precision default --repeats 9 --updates 10 --output outputs/saved-default-b128.json
