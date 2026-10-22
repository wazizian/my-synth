"""Bounded local Vast control; credentials stay outside the repository."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time

sys.path.insert(0, '/tmp/pilot-vast-access')
import vast_client as vast

ROOT = Path(__file__).resolve().parent
ARTIFACT_ROOT = ROOT / 'confirmation'
STATE = ROOT / 'instance.json'
LABEL = 'pilot-pallas-confirm-20261022'
LIFETIME = 3600
MAX_RATE = .60


def save(value):
    """Atomically persist campaign state dictionary; return nothing."""
    temp = STATE.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(STATE)


def own():
    """Return provider records bearing this study's unique label."""
    return [i for i in vast.list_instances() if i.get('label') == LABEL]


def destroy():
    """Delete only this study's instances and verify provider removal."""
    for instance in own():
        vast.api('DELETE', f"instances/{instance['id']}/")
    remaining = own()
    print(json.dumps({'remaining': [i['id'] for i in remaining]}), flush=True)
    if remaining:
        raise RuntimeError('Deletion not yet confirmed')
    if STATE.exists():
        state = json.loads(STATE.read_text())
        state['deleted_at'] = time.time()
        save(state)


def ssh():
    """Return direct SSH argv for the persisted, currently running study worker."""
    state = json.loads(STATE.read_text())
    instance = next(i for i in own() if i['id'] == state['id'])
    mappings = instance.get('ports') or {}
    port = (mappings.get('22/tcp') or [{}])[0].get('HostPort')
    host = instance.get('public_ipaddr')
    if not port or state.get('use_ssh_proxy'):
        port, host = instance.get('ssh_port'), instance.get('ssh_host')
    if not port or not host:
        raise RuntimeError('Worker SSH not ready; status=' + str(instance.get('actual_status')))
    return ['ssh', '-F', '/dev/null', '-i', str(vast.SSH), '-p', str(port),
            '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes',
            '-o', 'ConnectTimeout=15', '-o', 'StrictHostKeyChecking=accept-new',
            '-o', f'UserKnownHostsFile={ROOT / "known_hosts"}',
            'root@' + host]


def main():
    """Dispatch status/search/rent/watchdog/SSH/transfers with fixed budget boundaries."""
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['status', 'offers', 'rent', 'guard', 'watchdog', 'destroy', 'exec', 'upload', 'download', 'sync', 'monitor', 'monitor-service', 'new-attempt', 'stop-services'])
    parser.add_argument('arguments', nargs='*')
    args = parser.parse_args()
    if args.action == 'stop-services':
        subprocess.run(['systemctl', '--user', 'stop', 'pilot-pallas-watchdog', 'pilot-pallas-monitor'], check=False)
    elif args.action == 'new-attempt':
        old = json.loads(STATE.read_text())
        if any(i['id'] == old['id'] for i in vast.list_instances()):
            raise RuntimeError('Previous worker still exists; refusing replacement')
        history = ROOT / 'past-instances.json'
        entries = json.loads(history.read_text()) if history.exists() else []
        if not any(entry['id'] == old['id'] for entry in entries):
            entries.append(old)
        history.write_text(json.dumps(entries, indent=2))
        STATE.unlink()
        print('Archived deleted worker', old['id'])
    elif args.action == 'status':
        user = vast.api('GET', 'users/current/')
        print(json.dumps({'credit': user.get('credit'), 'instances': [
            {k: i.get(k) for k in ('id', 'label', 'actual_status', 'gpu_name', 'dph_total')}
            for i in vast.list_instances()]}, indent=2))
    elif args.action == 'offers':
        offers = vast.api('POST', 'bundles/', {
            'num_gpus': {'eq': 1}, 'gpu_ram': {'gte': 24000},
            'verified': {'eq': True}, 'rentable': {'eq': True}, 'rented': {'eq': False},
            'direct_port_count': {'gte': 1}, 'reliability2': {'gte': .98},
            'inet_down': {'gte': 1000}, 'cpu_cores_effective': {'gte': 6},
            'cuda_max_good': {'gte': 12.8}, 'dph_total': {'lte': MAX_RATE},
            'type': 'on-demand', 'allocated_storage': 30,
            'order': [['dph_total', 'asc']], 'limit': 80}).get('offers', [])
        (ROOT / 'offers.json').write_text(json.dumps(offers, indent=2))
        print(json.dumps([{k: o.get(k) for k in ('id', 'gpu_name', 'dph_total', 'inet_down', 'cpu_cores_effective', 'geolocation')} for o in offers], indent=2))
    elif args.action == 'rent':
        if STATE.exists() or own():
            raise RuntimeError('Existing study state; inspect before allocating again')
        offer = next(o for o in json.loads((ROOT / 'offers.json').read_text()) if o['id'] == int(args.arguments[0]))
        credit = float(vast.api('GET', 'users/current/')['credit'])
        rate = float(offer['dph_total'])
        if rate > MAX_RATE or credit < rate * LIFETIME / 3600 + 2:
            raise RuntimeError('Budget gate failed')
        state = {'label': LABEL, 'offer': offer['id'], 'created_at': time.time(),
                 'deadline': time.time() + LIFETIME, 'credit_before': credit,
                 'rate': rate, 'compute_cap': rate * LIFETIME / 3600}
        save(state)
        with (ROOT / 'watchdog.log').open('a') as log:
            process = subprocess.Popen([sys.executable, __file__, 'watchdog'],
                stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
        state['watchdog_pid'] = process.pid
        save(state)
        worker_guard = "import os,time,urllib.request; time.sleep(" + str(LIFETIME) + "); urllib.request.urlopen(urllib.request.Request('https://console.vast.ai/api/v0/instances/'+os.environ['CONTAINER_ID']+'/',method='DELETE',headers={'Authorization':'Bearer '+os.environ['CONTAINER_API_KEY']}),timeout=30).read()"
        result = vast.api('PUT', f"asks/{offer['id']}/", {
            'client_id': 'me', 'image': 'pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime',
            'disk': 30, 'label': LABEL, 'runtype': 'ssh_direct', 'cancel_unavail': True,
            'onstart': 'touch /root/.no_auto_tmux; nohup python -c ' + shlex.quote(worker_guard) + ' >/tmp/pallas-watchdog.log 2>&1 </dev/null &', 'env': {}})
        state['id'] = int(result['new_contract'])
        save(state)
        vast.api('POST', f"instances/{state['id']}/ssh/", {'ssh_key': Path(str(vast.SSH)+'.pub').read_text().strip()})
        print(json.dumps(state, indent=2))
    elif args.action == 'guard':
        subprocess.run(['systemd-run', '--user', '--unit=pilot-pallas-watchdog',
            '--collect', sys.executable, str(Path(__file__).resolve()), 'watchdog'], check=True)
        subprocess.run(['systemctl', '--user', 'show', 'pilot-pallas-watchdog',
                        '-p', 'ActiveState', '-p', 'MainPID'], check=True)
    elif args.action == 'monitor-service':
        subprocess.run(['systemd-run', '--user', '--unit=pilot-pallas-monitor',
            '--collect', sys.executable, str(Path(__file__).resolve()), 'monitor'], check=True)
        subprocess.run(['systemctl', '--user', 'show', 'pilot-pallas-monitor',
                        '-p', 'ActiveState', '-p', 'MainPID'], check=True)
    elif args.action == 'monitor':
        while not json.loads(STATE.read_text()).get('deleted_at'):
            subprocess.run([sys.executable, str(Path(__file__).resolve()), 'sync'], check=False, timeout=120)
            time.sleep(60)
    elif args.action == 'sync':
        result = subprocess.run(ssh() + ['cd /workspace/pallas; if test -d versions; then tar -czf - outputs versions; else tar -czf - outputs; fi'],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=90)
        if result.returncode not in (0, 1) or len(result.stdout) > 100_000_000:
            raise RuntimeError('Bounded artifact export failed')
        manifest = []
        with tarfile.open(fileobj=io.BytesIO(result.stdout), mode='r:gz') as archive:
            for member in archive.getmembers():
                path = Path(member.name)
                if not member.isfile():
                    continue
                if path.is_absolute() or '..' in path.parts or path.parts[0] not in ('outputs', 'versions'):
                    raise RuntimeError('Unsafe artifact path')
                data = archive.extractfile(member).read()
                destination = ARTIFACT_ROOT / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                temp = destination.with_name(destination.name+'.sync-tmp')
                temp.write_bytes(data)
                temp.replace(destination)
                manifest.append({'path': str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
        ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
        (ARTIFACT_ROOT / 'sync-manifest.json').write_text(json.dumps({'at': time.time(), 'files': manifest}, indent=2))
        print('Exported', len(manifest), 'files;', len(result.stdout), 'compressed bytes')
    elif args.action == 'watchdog':
        while time.time() < json.loads(STATE.read_text())['deadline']:
            if json.loads(STATE.read_text()).get('deleted_at'):
                return
            time.sleep(15)
        for attempt in range(30):
            try:
                destroy()
                return
            except Exception as error:
                print(type(error).__name__, str(error), flush=True)
                time.sleep(20)
        raise RuntimeError('Watchdog deletion failed')
    elif args.action == 'destroy':
        destroy()
    elif args.action == 'exec':
        subprocess.run(ssh() + [args.arguments[0]], check=True)
    elif args.action == 'upload':
        local, remote = args.arguments
        with open(local, 'rb') as stream:
            subprocess.run(ssh() + ['cat > ' + shlex.quote(remote)], stdin=stream, check=True)
    elif args.action == 'download':
        remote, local = args.arguments
        with open(local, 'wb') as stream:
            subprocess.run(ssh() + ['cat ' + shlex.quote(remote)], stdout=stream, check=True)


if __name__ == '__main__':
    main()
