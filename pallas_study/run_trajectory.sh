#!/bin/bash
set -euo pipefail
cd /workspace/pallas
trap 'code=$?; echo "$code" > outputs/trajectory.exit' EXIT
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 PYTHONPATH=/workspace/pallas JAX_PALLAS_VERBOSE_ERRORS=true
unset LD_LIBRARY_PATH
for ((attempt=0; attempt<12000; attempt++)); do
    test -f outputs/tensorcore.exit && break
    sleep 2
done
test -f outputs/tensorcore.exit
cp portable-persistent.py pallas_study/persistent.py
mkdir -p versions/v9
cp pallas_study/*.py versions/v9/
/workspace/pallas-env/bin/python -m pallas_study.validate_persistent
for batch in 128 1024; do
    /workspace/pallas-env/bin/python -m pallas_study.benchmark --batch "$batch" --steps 126 --persistent --analytic-jvp --precision default --tasks hedger historical joint --repeats 9 --updates 10 --output "outputs/persistent-default-b${batch}.json"
done
/workspace/pallas-env/bin/python -m pallas_study.benchmark --batch 128 --steps 126 --persistent --analytic-jvp --precision highest --dtype float64 --tasks historical --repeats 9 --updates 10 --output outputs/persistent-f64-b128.json
