#!/usr/bin/env python3
"""Rent one RTX 5090, smoke-test it over SSH, and destroy it."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import urllib.parse
import uuid

ROOT = Path(__file__).resolve().parent
STATE = ROOT / 'active.json'
KEY = Path.home() / 'vast_ai_api.txt'
SSH = Path.home() / '.ssh/id_ed25519_vastai'
BASE = 'https://console.vast.ai/api/v0/'


def api(method, path, body=None):
    req = urllib.request.Request(
        (BASE.replace('/v0/', '/v1/') if path.split('?')[0] == 'instances/' else BASE) + path, method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={'Authorization': 'Bearer ' + KEY.read_text().strip(),
                 'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=25) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        try:
            details = json.load(exc)
            message = str(details.get('msg') or details.get('error') or '')[:400]
        except (ValueError, AttributeError):
            message = ''
        raise RuntimeError(f'Vast API HTTP {exc.code} on {method} {path}: {message}') from None
    except urllib.error.URLError:
        raise RuntimeError('Cannot reach Vast.ai API (network/DNS unavailable).') from None
    if isinstance(result, dict) and result.get('success') is False:
        if path.endswith('/ssh/') and result.get('msg') == 'SSH key already associated with instance.':
            return result
        raise RuntimeError(f'Vast API rejected {method} {path}')
    return result


def list_instances():
    instances = []
    path = 'instances/'
    while True:
        page = api('GET', path)
        instances.extend(page['instances'])
        token = page.get('next_token')
        if not token:
            return instances
        path = 'instances/?' + urllib.parse.urlencode({'after_token': token})


def _unsafe_cleanup(expected_label=None):
    if not STATE.exists():
        return
    saved = json.loads(STATE.read_text())
    if expected_label is not None and saved['label'] != expected_label:
        return
    for attempt in range(6):
        try:
            instances = list_instances()
            targets = [x for x in instances if x.get('label') == saved['label']]
            for item in targets:
                if item.get('actual_status') != 'destroyed':
                    api('DELETE', f"instances/{item['id']}/")
                    print(f"Destroy requested for {item['id']}", flush=True)
            remaining = list_instances()
            if not any(x.get('label') == saved['label'] and
                       x.get('actual_status') != 'destroyed' for x in remaining):
                STATE.unlink(missing_ok=True)
                print('Confirmed: no active smoke-test instance remains.', flush=True)
                return
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr, flush=True)
        time.sleep(10)
    raise RuntimeError(f'Cleanup unconfirmed! Run {__file__} --cleanup; state: {STATE}')


# VERIFIED_EXPORT_GUARD
def cleanup(expected_label=None):
    from export_safety import guarded_cleanup
    return guarded_cleanup(sys.modules[__name__], expected_label)

def run(args, workload=None):
    if STATE.exists():
        raise RuntimeError('Previous run needs cleanup. Run with --cleanup first.')
    for path in (KEY, SSH, Path(str(SSH) + '.pub')):
        if not path.is_file():
            raise RuntimeError(f'Missing credential file: {path}')
    query = {'gpu_name': {'eq': 'RTX 5090'}, 'num_gpus': {'eq': 1},
             'gpu_ram': {'gte': 24000}, 'verified': {'eq': True},
             'rentable': {'eq': True}, 'rented': {'eq': False},
             'external': {'eq': False}, 'direct_port_count': {'gte': 1},
             'reliability2': {'gte': .98}, 'type': 'on-demand',
             'allocated_storage': getattr(args, 'disk', 20),
             'order': [['dph_total', 'asc']], 'limit': 100}
    if workload:
        query['cuda_max_good'] = {'gte': 13.0}
        query['inet_down'] = {'gte': 500}
    offers = api('POST', 'bundles/', query).get('offers', [])
    stale_path = Path('/home/jump/campaign-control/stale_offer_ids.json')
    stale_ids = set(json.loads(stale_path.read_text())) if stale_path.exists() else set()
    offers = [offer for offer in offers if offer.get('id') not in stale_ids]
    excluded_ids = set(getattr(args, 'exclude_offer', []))
    offers = [offer for offer in offers if offer.get('id') not in excluded_ids]
    offers = [o for o in offers if float(o['dph_total']) <= args.max_hourly]
    if not offers:
        raise RuntimeError(f'No qualifying RTX 5090 offer below ${args.max_hourly:.2f}/hour.')
    offer = min(offers, key=lambda o: float(o['dph_total']))
    print(f"Offer {offer['id']}: RTX 5090, {offer['gpu_ram']} MB, "
          f"listed ${offer['dph_total']:.4f}/hour (bandwidth may be extra)", flush=True)
    if args.search_only:
        return
    label = 'codex-4090-smoke-' + uuid.uuid4().hex
    with STATE.open('x') as state_file:
        json.dump({'label': label, 'offer': offer['id']}, state_file)
    os.chmod(STATE, 0o600)
    # Systemd global watchdog owns deadlines; no unverified local cleanup child.
    try:
        # FUNDING_CREATE_BARRIER
        if Path('/home/jump/deep-hedging/campaigns/comparison-20261018/funding_halt.json').exists():
            raise RuntimeError('Durable funding halt blocks provider create')
        created = api('PUT', f"asks/{offer['id']}/", {
            'client_id': 'me', 'image': getattr(args, 'image', 'pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime'),
            'disk': getattr(args, 'disk', 20), 'label': label, 'runtype': 'ssh_direct',
            'onstart': 'touch /root/.no_auto_tmux',
            'cancel_unavail': True, 'env': {}})
        instance_id = int(created['new_contract'])
        saved = json.loads(STATE.read_text())
        saved.update(instance_id=instance_id, listed_hourly=offer['dph_total'])
        STATE.write_text(json.dumps(saved))
        print(f'Created instance {instance_id}', flush=True)
        api('POST', f'instances/{instance_id}/ssh/',
            {'ssh_key': Path(str(SSH) + '.pub').read_text().strip()})
        deadline = time.monotonic() + getattr(args, 'boot_timeout_seconds', 300)
        while time.monotonic() < deadline:
            instances = list_instances()
            instance = next((x for x in instances if x['id'] == instance_id), {})
            if not STATE.exists():
                raise RuntimeError('Instance was cleaned up by another process.')
            print('Status:', instance.get('actual_status', 'pending'), flush=True)
            mappings = instance.get('ports') or {}
            port = (mappings.get('22/tcp') or [{}])[0].get('HostPort')
            host = instance.get('public_ipaddr')
            if instance.get('actual_status') == 'running' and host and port:
                command = ['ssh', '-F', '/dev/null', '-i', str(SSH), '-p', str(port),
                           '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes',
                           '-o', 'ConnectTimeout=10', '-o', 'StrictHostKeyChecking=accept-new',
                           '-o', f'UserKnownHostsFile={ROOT / "known_hosts"}', f'root@{host}']
                ready = subprocess.run(command + ['true'], capture_output=True, timeout=15)
                if ready.returncode == 0:
                    if workload:
                        workload(command)
                        return
                    remote = """nvidia-smi --query-gpu=name,memory.total --format=csv
python - <<'PY'
import time, torch
assert torch.cuda.is_available(), 'CUDA unavailable'
assert torch.cuda.device_count() == 1
p = torch.cuda.get_device_properties(0)
assert '4090' in p.name and p.total_memory >= 23 * 1024**3, p
x = torch.randn((2048, 2048), device='cuda')
start = time.monotonic()
count = 0
while time.monotonic() - start < 15:
    y = x @ x
    torch.cuda.synchronize()
    count += 1
assert torch.isfinite(y).all().item()
print(f'PASS: {p.name}, {p.total_memory / 1024**3:.1f} GiB, {count} GPU matmuls in 15 seconds')
PY"""
                    subprocess.run(command + [remote], check=True, timeout=60)
                    return
            time.sleep(10)
        raise RuntimeError('Instance did not become SSH-ready before the boot timeout.')
    finally:
        cleanup(label)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-hourly', type=float, default=.60)
    parser.add_argument('--search-only', action='store_true')
    parser.add_argument('--cleanup', action='store_true')
    parser.add_argument('--delay', type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument('--expected-label', help=argparse.SUPPRESS)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        if args.cleanup:
            time.sleep(args.delay)
            if args.expected_label and (not STATE.exists() or
                    json.loads(STATE.read_text()).get('label') != args.expected_label):
                return 0
            cleanup(args.expected_label)
        else:
            run(args)
    except (RuntimeError, OSError, subprocess.SubprocessError, KeyError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
