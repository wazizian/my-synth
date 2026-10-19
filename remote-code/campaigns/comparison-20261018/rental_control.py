"""Bounded retries only before any workload-launch intent, with provider absence."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from controller_handoff import atomic_json
from offer_exclusions import exclusion_arguments
sys.path.insert(0, '/home/jump/vast-tools')
import smoke


def confirm_failed_absence(launcher, log_path, state):
    if (launcher / 'active.json').exists():
        raise RuntimeError('Failed allocation has active instance; reattach, never retry')
    runs = list((launcher / 'runs').glob('*'))
    if any((run / 'workload-launch-intent.json').exists() for run in runs):
        raise RuntimeError('Failed attempt has workload-launch intent; scientific retry refused')
    ids = set()
    for run in runs:
        metadata = run / 'metadata.json'
        if metadata.exists():
            ids.add(json.loads(metadata.read_text())['instance']['instance_id'])
    text = log_path.read_text() if log_path.exists() else ''
    offset = state.get('last_attempt_log_offset')
    if type(offset) is not int or offset < 0:
        raise RuntimeError('Missing bounded attempt log provenance; root review required')
    with log_path.open('rb') as stream:
        stream.seek(offset);attempt_text = stream.read().decode(errors='replace')
    errors = [re.split(r'(?:RuntimeError:|ERROR:)\s*', line, maxsplit=1)[-1] for line in attempt_text.splitlines() if re.search(r'(?:RuntimeError:|ERROR:)\s*', line)]
    error = errors[-1] if errors else attempt_text.strip().splitlines()[-1] if attempt_text.strip() else ''
    if error.startswith('No qualifying RTX 4090 offer below '):
        classification = 'no_qualifying_offer_no_create'
    else:
        rejected = re.match(r'Vast API HTTP (400|404|409) on PUT asks/\d+/: (.+)', error)
        if not rejected or not any(word in rejected.group(2).lower() for word in ('not available', 'no_such_ask', 'not found', 'does not exist', 'already rented')):
            raise RuntimeError('Ambiguous or non-preallocation failure; root review required')
        classification = 'definitive_unavailable_ask_rejection'
    state['preallocation_failure_classification'] = classification
    ids.update(int(value) for value in re.findall(r'Created instance (\d+)', text))
    live = {row['id'] for row in smoke.list_instances()}
    if live - set(state['last_attempt_before_provider_ids']):
        raise RuntimeError('New provider instance identity uncertain after failure; retry refused')
    if ids & live:
        raise RuntimeError('Prior allocation remains at provider; retry refused')
    state['attempt_status'] = 'failed_provider_absent_no_intent'
    state['excluded_offers'] = sorted(set(state['excluded_offers']) | {
        int(value) for value in re.findall(r'Offer (\d+):', text)})
    state['prior_instance_ids_confirmed_absent'] = sorted(ids)
    state['provider_absence_query_utc'] = datetime.now(timezone.utc).isoformat()


def resumable_preallocation(launcher, log_path):
    launcher = Path(launcher)
    path = launcher / 'preallocation-retries.json'
    if not path.exists():
        return False
    state = json.loads(path.read_text())
    if (state.get('attempt_status') not in ('returned_nonzero', 'failed_provider_absent_no_intent')
            or type(state.get('child_returncode')) is not int or state['child_returncode'] == 0
            or type(state.get('attempts_started')) is not int
            or not 1 <= state['attempts_started'] <= 3
            or not isinstance(state.get('last_attempt_before_provider_ids'), list)
            or any(type(value) is not int for value in state['last_attempt_before_provider_ids'])):
        return False
    confirm_failed_absence(launcher, Path(log_path), state)
    atomic_json(path, state)
    return True


def run_preallocation(command, launcher, log_path, deadline):
    launcher, log_path = Path(launcher), Path(log_path)
    state_path = launcher / 'preallocation-retries.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        'attempts_started': 0, 'retries': 0, 'excluded_offers': []}
    if state.get('attempt_status') == 'completed':
        return
    while True:
        remaining = int(deadline - time.time() - 300)
        if remaining <= 0:
            raise RuntimeError('Allocation deadline exhausted; no rental or retry')
        # Absence cannot disambiguate a provider create request still in flight.
        if state.get('attempt_status') == 'inflight':
            raise RuntimeError('Interrupted allocation lacks durable child exit; root review required')
        if state.get('attempt_status') in ('returned_nonzero', 'failed_provider_absent_no_intent') and (type(state.get('child_returncode')) is not int or state['child_returncode'] == 0):
            raise RuntimeError('Persisted failed allocation lacks strict nonzero child exit receipt')
        if state.get('attempt_status') == 'returned_nonzero':
            confirm_failed_absence(launcher, log_path, state)
            atomic_json(state_path, state)
        if state['attempts_started'] >= 3:
            raise RuntimeError('Two bounded preallocation infrastructure retries exhausted')
        actual = list(command)
        if actual[:4] == ['uv', 'run', '--no-project', 'python']:
            actual = [sys.executable] + actual[4:]
        actual += exclusion_arguments()
        for offer in state['excluded_offers']:
            actual += ['--exclude-offer', str(offer)]
        if '--timeout-seconds' in actual:
            index = actual.index('--timeout-seconds') + 1
            actual[index] = str(min(int(actual[index]), remaining))
        state['last_attempt_before_provider_ids'] = sorted(row['id'] for row in smoke.list_instances())
        state['last_attempt_started_utc'] = datetime.now(timezone.utc).isoformat()
        state['attempts_started'] += 1
        state['retries'] = state['attempts_started'] - 1
        state['attempt_status'] = 'inflight'
        state['last_attempt_log_offset'] = log_path.stat().st_size if log_path.exists() else 0
        atomic_json(state_path, state)
        with log_path.open('a') as log:
            result = subprocess.run(actual, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode == 0:
            state['attempt_status'] = 'completed'
            atomic_json(state_path, state)
            return
        state['attempt_status'] = 'returned_nonzero'
        state['child_returncode'] = result.returncode
        atomic_json(state_path, state)
        confirm_failed_absence(launcher, log_path, state)
        atomic_json(state_path, state)
