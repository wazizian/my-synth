"""Bounded transfer of a live remote run to the durable recovery service."""
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess
import time


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def request_handoff(run_dir, ssh, args):
    metadata = json.loads((run_dir / 'metadata.json').read_text())
    # A separate connection proves the detached supervisor survived launch SSH.
    result = subprocess.run(ssh + ["pgrep -af '^bash supervise-workload.sh$'"],
                            capture_output=True, text=True, timeout=45, check=True)
    if not result.stdout.strip():
        raise RuntimeError('Detached supervisor is not alive')
    request = dict(run_id=run_dir.name,
                   instance_id=metadata['instance']['instance_id'],
                   label=metadata['instance']['label'], ssh=ssh,
                   required_artifacts=args.require_artifact,
                   deadline_utc=args.handoff_deadline_utc,
                   supervisor=result.stdout.strip(),
                   source_sha256=hashlib.sha256((run_dir / 'submitted-job.py').read_bytes()).hexdigest(),
                   created_utc=datetime.now(timezone.utc).isoformat())
    request_path = run_dir / 'handoff.request.json'
    atomic_json(request_path, request)
    digest = hashlib.sha256(request_path.read_bytes()).hexdigest()
    deadline = time.monotonic() + 180
    ack_path = run_dir / 'handoff.ack.json'
    while time.monotonic() < deadline:
        if ack_path.exists():
            ack = json.loads(ack_path.read_text())
            if (ack.get('request_sha256') == digest and ack.get('owner') == 'dh-credit-recovery'
                    and ack.get('instance_id') == request['instance_id']
                    and ack.get('run_id') == request['run_id']
                    and ack.get('deadline_utc') == request['deadline_utc']
                    and ack.get('required_artifacts') == request['required_artifacts']):
                atomic_json(run_dir / 'handoff.accepted.json', ack)
                return
            raise RuntimeError('Recovery ownership ACK does not match this run')
        time.sleep(2)
    raise RuntimeError('Durable recovery ownership ACK timed out')
