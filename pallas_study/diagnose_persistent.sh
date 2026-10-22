#!/bin/bash
set -euo pipefail
cd /workspace/pallas
trap 'code=$?; echo "$code" > outputs/persistent-diagnostic.exit' EXIT
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 PYTHONPATH=/workspace/pallas JAX_PALLAS_VERBOSE_ERRORS=true
unset LD_LIBRARY_PATH
for ((attempt=0; attempt<12000; attempt++)); do
    test -f outputs/final.exit && break
    sleep 2
done
test -f outputs/final.exit
/workspace/pallas-env/bin/python -m pallas_study.validate_persistent
