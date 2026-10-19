"""Durable bounded stdlib proof harness; test sidecars never own worker lifetime."""
import json
import os
from pathlib import Path
import time
output = Path(os.environ['JOB_OUTPUT_DIR'])
start = time.monotonic()
while time.monotonic() - start < 1650:
    temp = output / 'control-heartbeat.json.tmp'
    temp.write_text(json.dumps({'elapsed_seconds': time.monotonic()-start, 'pid': os.getpid()}))
    temp.replace(output / 'control-heartbeat.json')
    if (output / 'RELEASE').exists():
        (output / 'proof-worker-complete.json').write_text(json.dumps({'released': True}))
        break
    time.sleep(5)
else:
    raise TimeoutError('Control proof harness timed out without release')
