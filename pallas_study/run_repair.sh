#!/bin/bash
set -uo pipefail
cd /workspace/pallas
trap 'code=$?; echo "$code" > outputs/repair.exit' EXIT
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 PYTHONPATH=/workspace/pallas
unset LD_LIBRARY_PATH
for ((attempt=0; attempt<12000; attempt++)); do
    test -f outputs/default.exit && break
    sleep 2
done
test -f outputs/default.exit || exit 1
cp repaired-kernels.py pallas_study/kernels.py
cp repaired-persistent.py pallas_study/persistent.py
cp repaired-meta.py pallas_study/meta_benchmark.py
mkdir -p versions/v7
cp pallas_study/*.py versions/v7/
/workspace/pallas-env/bin/python -m pallas_study.validate_persistent
code=$?
echo "$code" > outputs/persistent-validation.exit
if test "$code" = 0; then
    for batch in 128 1024; do
        /workspace/pallas-env/bin/python -m pallas_study.benchmark --batch "$batch" --steps 126 --persistent --analytic-jvp --precision default --tasks hedger historical joint --repeats 9 --updates 10 --output "outputs/persistent-default-b${batch}.json"
    done
    /workspace/pallas-env/bin/python -m pallas_study.benchmark --batch 128 --steps 126 --persistent --analytic-jvp --precision highest --dtype float64 --tasks historical --repeats 9 --updates 10 --output outputs/persistent-f64-b128.json
fi
/workspace/pallas-env/bin/python -m pallas_study.meta_benchmark --output outputs/meta-v7.json
echo "$?" > outputs/meta-v7.exit
for task in hedger sde; do
    /workspace/pallas-env/bin/python -m pallas_study.credit_benchmark --task "$task" --analytic-only --output "outputs/credit-${task}-analytic.json"
    echo "$?" > "outputs/credit-${task}-analytic.exit"
done
