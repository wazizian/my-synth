"""Short worker-only durability probe; no local execution."""
import json
import os
from pathlib import Path
import time

output = Path(os.environ['JOB_OUTPUT_DIR'])
for step in range(12):
    temporary = output / 'progress.json.tmp'
    temporary.write_text(json.dumps(dict(step=step, pid=os.getpid(), time=time.time())))
    temporary.replace(output / 'progress.json')
    time.sleep(10)
temporary = output / 'complete.json.tmp'
temporary.write_text(json.dumps(dict(status='complete', pid=os.getpid(), time=time.time())))
temporary.replace(output / 'complete.json')
