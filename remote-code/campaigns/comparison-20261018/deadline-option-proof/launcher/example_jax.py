from pathlib import Path
import os
Path(os.environ['JOB_OUTPUT_DIR']).mkdir(parents=True, exist_ok=True)
Path(os.environ['JOB_OUTPUT_DIR'], 'stdlib-control-only.txt').write_text('No scientific preflight performed.\n')
