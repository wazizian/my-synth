#!/bin/bash
set -euo pipefail
cd /workspace/pallas
trap 'code=$?; echo "$code" > outputs/persistent.exit' EXIT
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 PYTHONPATH=/workspace/pallas
unset LD_LIBRARY_PATH
for ((attempt=0; attempt<6000; attempt++)); do
    test -f outputs/saved.exit && break
    sleep 2
done
test "$(cat outputs/saved.exit)" = 0
cp trajectory-benchmark.py pallas_study/benchmark.py
mkdir -p versions/v3
cp pallas_study/*.py versions/v3/
/workspace/pallas-env/bin/python -m pallas_study.validate_persistent
for batch in 128 256 1024; do
    /workspace/pallas-env/bin/python -m pallas_study.benchmark --batch "$batch" --steps 126 --saved-backward --persistent --precision highest --tasks hedger historical joint --repeats 9 --updates 10 --output "outputs/persistent-high-b${batch}.json"
done
/workspace/pallas-env/bin/python -m pallas_study.benchmark --batch 128 --steps 126 --saved-backward --persistent --precision highest --dtype float64 --tasks historical --repeats 9 --updates 10 --output outputs/persistent-f64-b128.json
