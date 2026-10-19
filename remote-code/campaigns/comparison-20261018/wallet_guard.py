"""Independent financial stop barrier; preserves every unverified worker disk."""
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
from controller_handoff import atomic_json
sys.path.insert(0, '/home/jump/vast-tools')
import smoke
ROOT = Path('/home/jump/deep-hedging/campaigns/comparison-20261018')
RESERVE_USD = 2.5


def assert_funding_available():
    if (ROOT / 'funding_halt.json').exists():
        raise RuntimeError('Durable funding halt blocks new allocation; root funding review required')


def monitor_wallet(instances, launches):
    state_path = ROOT / 'funding-wallet-state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if time.time() - state.get('last_attempt_epoch', 0) < 60:
        return
    state['last_attempt_epoch'] = time.time()
    state['query_utc'] = datetime.now(timezone.utc).isoformat()
    halt_path = ROOT / 'funding_halt.json'
    try:
        wallet = smoke.api('GET', 'users/current')
        credit = wallet.get('credit', wallet.get('balance'))
        if type(credit) not in (int, float) or not math.isfinite(credit):
            raise RuntimeError('Provider wallet has no finite numeric available-credit field')
        if instances is None:
            instances = smoke.list_instances()
        rate = sum(float(row.get('dph_total') or 0) for row in instances
                   if row.get('actual_status') not in ('stopped', 'destroyed'))
        buffer = max(.5, rate / 6)  # Ten minutes of observed listed burn.
        state.update(consecutive_failures=0, provider_available_credit_usd=credit,
                     wallet_fields={key: wallet[key] for key in ('credit', 'balance') if key in wallet},
                     current_listed_usd_per_hour=rate, protected_reserve_usd=RESERVE_USD,
                     bounded_burn_buffer_usd=buffer, stop_threshold_usd=RESERVE_USD + buffer)
        if credit <= RESERVE_USD + buffer:
            halt = dict(reason='available_credit_below_protected_reserve_and_burn_buffer',
                        compute_stop_required=True, root_review_required=True, **state)
            atomic_json(halt_path, halt)
        elif halt_path.exists() and json.loads(halt_path.read_text()).get('reason') == 'wallet_unavailable_after_three_queries':
            halt_path.unlink()  # Recovery from API outage is safe only with a fresh healthy wallet.
    except Exception as error:
        state['consecutive_failures'] = state.get('consecutive_failures', 0) + 1
        state['last_error'] = repr(error)
        if state['consecutive_failures'] >= 3 and not halt_path.exists():
            atomic_json(halt_path, dict(reason='wallet_unavailable_after_three_queries',
                                       compute_stop_required=False, root_review_required=False, **state))
    atomic_json(state_path, state)
    if not halt_path.exists():
        return
    halt = json.loads(halt_path.read_text())
    if not halt.get('compute_stop_required'):
        return
    # Identity comes only from durable launcher metadata and fresh provider rows.
    if instances is None:
        try:instances = smoke.list_instances()
        except Exception as error:
            halt['stop_provider_query_error'] = repr(error)
            atomic_json(halt_path, halt)
            return
    targets = []
    for launch in launches:
        active = Path(launch) / 'active.json'
        if not active.exists():
            continue
        saved = json.loads(active.read_text())
        matching = next((row for row in instances if row.get('id') == saved.get('instance_id')
                         and row.get('label') == saved.get('label')), None)
        if matching is None:
            continue
        target = dict(instance_id=saved['instance_id'], label=saved['label'],
                      launch=str(launch), source_disk_preservation_requested=True, deletion_requested=False)
        targets.append(target)
        # Persist target intent before potentially interrupted provider request.
        halt['stop_targets'] = targets
        atomic_json(halt_path, halt)
        if matching.get('actual_status') != 'stopped':
            try:
                smoke.api('PUT', f"instances/{saved['instance_id']}/", {'state': 'stopped'})
                target['stop_requested'] = True
            except Exception as error:
                target['stop_error'] = repr(error)
    try:
        refreshed = smoke.list_instances()
        for target in targets:
            matching = next((row for row in refreshed if row.get('id') == target['instance_id']
                             and row.get('label') == target['label']), None)
            target['compute_stop_confirmed'] = matching is None or matching.get('actual_status') == 'stopped'
            target['disk_retained_confirmed'] = matching is not None and matching.get('actual_status') == 'stopped'
    except Exception as error:
        halt['stop_confirmation_error'] = repr(error)
    halt['stop_targets'] = targets
    halt['last_stop_attempt_utc'] = datetime.now(timezone.utc).isoformat()
    atomic_json(halt_path, halt)


def wallet_loop(discover):
    """Separate thread keeps financial polling alive during bounded exports."""
    while True:
        try:
            monitor_wallet(None, discover())
        except Exception as error:
            atomic_json(ROOT / 'funding-guard-error.json', dict(error=repr(error), utc=datetime.now(timezone.utc).isoformat()))
        time.sleep(5)
