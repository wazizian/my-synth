#!/bin/bash
set -euo pipefail
cd /workspace/pallas
trap 'code=$?; echo "$code" > outputs/persistent-retry.exit' EXIT
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 PYTHONPATH=/workspace/pallas
unset LD_LIBRARY_PATH
for ((attempt=0; attempt<11000; attempt++)); do
    test -f outputs/analytic.exit && break
    sleep 2
done
test -f outputs/analytic.exit
mkdir -p versions/v5
cp pallas_study/*.py versions/v5/
/workspace/pallas-env/bin/python -m pallas_study.validate_persistent
for batch in 128 256 1024; do
    /workspace/pallas-env/bin/python -m pallas_study.benchmark --batch "$batch" --steps 126 --analytic-jvp --persistent --precision highest --tasks hedger historical joint --repeats 9 --updates 10 --output "outputs/persistent-high-b${batch}.json"
done
/workspace/pallas-env/bin/python -m pallas_study.benchmark --batch 128 --steps 126 --analytic-jvp --persistent --precision default --tasks hedger historical joint --repeats 9 --updates 10 --output outputs/persistent-default-b128.json
/workspace/pallas-env/bin/python -m pallas_study.benchmark --batch 128 --steps 126 --analytic-jvp --persistent --precision highest --dtype float64 --tasks historical --repeats 9 --updates 10 --output outputs/persistent-f64-b128.json
