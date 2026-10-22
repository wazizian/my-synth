#!/bin/bash
set -euo pipefail
cd /workspace/pallas
trap 'code=$?; echo "$code" > outputs/saved.exit' EXIT
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 PYTHONPATH=/workspace/pallas
unset LD_LIBRARY_PATH
for ((attempt=0; attempt<3000; attempt++)); do
    test -f outputs/full.exit && break
    sleep 2
done
test "$(cat outputs/full.exit)" = 0
mkdir -p versions/v1 versions/v2
cp pallas_study/kernels.py pallas_study/benchmark.py versions/v1/
cp next-kernels.py pallas_study/kernels.py
cp next-benchmark.py pallas_study/benchmark.py
cp pallas_study/kernels.py pallas_study/benchmark.py versions/v2/
/workspace/pallas-env/bin/python -m pallas_study.benchmark --batch 128 --steps 126 --saved-backward --precision highest --repeats 9 --updates 10 --output outputs/saved-high-b128.json
/workspace/pallas-env/bin/python -m pallas_study.benchmark --batch 128 --steps 126 --precision highest --repeats 9 --updates 10 --output outputs/jvp-high-b128.json
