#!/bin/bash
set -euo pipefail
cd /workspace/pallas
trap 'code=$?; echo "$code" > outputs/benchmark.exit' EXIT
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
export PYTHONPATH=/workspace/pallas
unset LD_LIBRARY_PATH
for ((attempt=0; attempt<450; attempt++)); do
    test -f outputs/setup.done && break
    sleep 2
done
test -f outputs/setup.done
/workspace/pallas-env/bin/python -m pallas_study.benchmark --batch 32 --steps 8 --repeats 3 --updates 3 --output outputs/smoke.json
