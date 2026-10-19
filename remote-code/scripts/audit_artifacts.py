"""Metadata-only archive completeness audit; safe on the control plane."""
import argparse
import json
from pathlib import Path

ARMS = ('baseline', 'hard_reference', 'dpg_hard', 'soar_hard', 'selfplay', 'dpg_mixed', 'soar_mixed')
DOMAINS = ('base', 'easy', 'hard')


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def read_events(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def expand_event(event):
    """Accept one domain/event or a domain-mapped metrics payload."""
    payload = event.get('metrics', {})
    if isinstance(payload, dict) and any(k in payload for k in DOMAINS):
        return [dict(event, **value, domain=domain) for domain, value in payload.items()
                if domain in DOMAINS and isinstance(value, dict)]
    if isinstance(payload, dict):
        return [dict(event, **payload)]
    return [event]


def selected_runs(manifest):
    selected = (manifest or {}).get('selected', (manifest or {}).get('runs', {}))
    if isinstance(selected, list):
        return {row.get('method', row.get('arm', row.get('run_id'))): row for row in selected}
    return {arm: ({'run_id': row} if isinstance(row, str) else row) for arm, row in selected.items()}


def audit(root):
    root = Path(root)
    failures, runs = [], []
    manifest = read_events(root / 'runs/manifest.jsonl')
    manifest_ids = {e.get('run_id', e.get('trial_id')) for e in manifest}
    directories = sorted(p for p in (root / 'runs').glob('*') if p.is_dir())
    for path in directories:
        issues = []
        config = read_json(path / 'config.json', {})
        status = read_json(path / 'status.json', {})
        train = read_events(path / 'train_metrics.jsonl')
        events = [row for event in read_events(path / 'evaluation_metrics.jsonl') for row in expand_event(event)]
        if not config: issues.append('missing config')
        if not status: issues.append('missing status')
        if not train: issues.append('missing training events')
        if path.name not in manifest_ids: issues.append('missing manifest entry')
        for domain in DOMAINS:
            if not any(e.get('domain') == domain and e.get('split_role') == 'monitor' for e in events):
                issues.append('missing monitor: ' + domain)
        bundles = list((path / 'checkpoints').glob('*/metadata.json'))
        reason = status.get('reason', status.get('error', status.get('exception')))
        if not bundles and not reason: issues.append('missing checkpoint and failure reason')
        for bundle in bundles:
            meta = read_json(bundle, {})
            checkpoint_id = meta.get('checkpoint_id', bundle.parent.name)
            for domain in DOMAINS:
                matching = [e for e in events if e.get('checkpoint_id') == checkpoint_id
                            and e.get('domain') == domain and e.get('split_role') == 'final_test']
                if not matching:
                    issues.append('missing final_test: ' + checkpoint_id + '/' + domain)
                elif not any(e.get('raw_mse') is not None or e.get('reason') or e.get('error') for e in matching):
                    issues.append('null final_test without reason: ' + checkpoint_id + '/' + domain)
            if not (bundle.parent / 'state.eqx').exists(): issues.append('missing state: ' + checkpoint_id)
        if bundles and not any((path / 'checkpoints' / (tag + '.json')).exists() for tag in ('latest', 'final')):
            issues.append('missing terminal/latest pointer')
        runs.append(dict(run_id=path.name, method=config.get('method'), issues=issues, status=status))
        failures.extend(path.name + ': ' + issue for issue in issues)
    selection = selected_runs(read_json(root / 'results/frozen_selection.json', {}))
    for arm in ARMS:
        if arm not in selection: failures.append('missing selected arm: ' + arm)
    for run_id in manifest_ids - {p.name for p in directories} - {None}:
        failures.append('manifest run missing directory: ' + run_id)
    return dict(complete=not failures, scope='metadata only; array integrity requires GPU-worker validation',
                runs=runs, failures=failures, selected=selection)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit(args.root)
    output = args.output or args.root / 'results/artifact_audit.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'complete': result['complete'], 'failures': result['failures']}))
    raise SystemExit(0 if result['complete'] else 1)
