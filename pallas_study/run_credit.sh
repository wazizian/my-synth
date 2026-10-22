#!/bin/bash
set -uo pipefail
cd /workspace/pallas
trap 'code=$?; echo "$code" > outputs/credit.exit' EXIT
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 PYTHONPATH=/workspace/pallas
unset LD_LIBRARY_PATH
for ((attempt=0; attempt<9000; attempt++)); do
    test -f outputs/persistent.exit && break
    sleep 2
done
test -f outputs/persistent.exit || exit 1
/workspace/pallas-env/bin/python -m pallas_study.benchmark --batch 128 --steps 126 --saved-backward --persistent --precision highest --dtype float64 --tasks historical --repeats 9 --updates 10 --output outputs/persistent-f64-b128.json
echo "$?" > outputs/historical-f64.exit
for task in hedger sde; do
    /workspace/pallas-env/bin/python -m pallas_study.credit_benchmark --task "$task" --output "outputs/credit-${task}.json"
    echo "$?" > "outputs/credit-${task}.exit"
done
