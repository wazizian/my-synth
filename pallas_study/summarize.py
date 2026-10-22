"""Build an auditable timing table from completed GPU result files."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent


def main():
    """Write summary CSV/JSON with correctness gates and median whole-update speed ratios."""
    rows = []
    files = sorted((ROOT / 'outputs').glob('*.json'))+sorted((ROOT / 'confirmation' / 'outputs').glob('*.json'))
    for path in files:
        report = json.loads(path.read_text())
        if not report.get('complete') or path.name == 'smoke.json' or path.name.startswith('meta'):
            continue
        config = report.get('config', {})
        baselines = {entry.get('task', report.get('task')): entry for entry in report['results'] if entry['variant'] == 'jax'}
        for entry in report['results']:
            task = entry.get('task', report.get('task'))
            baseline = baselines[task]
            timings = entry.get('update_ms', [])
            baseline_timings = baseline.get('update_ms', [])
            median = statistics.median(timings) if timings else None
            baseline_median = statistics.median(baseline_timings) if baseline_timings else None
            trajectory_error = entry.get('parameters_after_20_updates', entry.get('parameter_error_after_10_updates'))
            final_key = 'loss_after_20_updates' if 'loss_after_20_updates' in entry else 'loss_after_10_updates'
            final_error = abs(entry[final_key]-baseline[final_key]) if final_key in entry and final_key in baseline else None
            trajectory_ok = (trajectory_error is not None and final_error is not None and final_error < 2e-5 and
                (trajectory_error['relative_l2'] < .001 or trajectory_error['max_abs'] < .0001))
            row = {'file': str(path.relative_to(ROOT)), 'device_kind': report.get('device_kind', 'A100' if 'confirmation' in path.parts else 'RTX 4090'), 'task': task, 'variant': entry['variant'],
                   'batch': config.get('batch', report.get('batch')), 'steps': config.get('steps', report.get('steps')),
                   'dtype': config.get('dtype', report.get('dtype', 'float32')),
                   'precision': config.get('precision', 'highest' if 'high' in path.name or path.name.startswith('credit-') else 'default'),
                   'saved_backward': config.get('saved_backward', entry['variant'] == 'pallas_saved'),
                   'analytic_jvp': config.get('analytic_jvp', False),
                   'initial_correct': entry.get('correct', False), 'trajectory_correct': trajectory_ok,
                   'eligible': bool(entry.get('correct') and trajectory_ok),
                   'median_update_ms': median, 'baseline_update_ms': baseline_median,
                   'speedup': baseline_median/median if median and baseline_median else None,
                   'min_update_ms': min(timings) if timings else None,
                   'max_update_ms': max(timings) if timings else None,
                   'gradient_max_abs': entry.get('gradient_error', {}).get('max_abs'),
                   'gradient_relative_l2': entry.get('gradient_error', {}).get('relative_l2'),
                   'final_loss_error': final_error,
                   'parameter_relative_l2': trajectory_error['relative_l2'] if trajectory_error else None,
                   'compile_and_first_update_s': entry.get('compile_and_first_update_s'),
                   'error': entry.get('error')}
            rows.append(row)
    (ROOT / 'summary.json').write_text(json.dumps(rows, indent=2))
    if rows:
        with (ROOT / 'summary.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    manifest = []
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or path.name in ('known_hosts', 'instance.json', 'offers.json', 'sync-manifest.json', 'manifest.json'):
            continue
        if path.suffix not in ('.py', '.sh', '.json', '.csv', '.txt', '.md', '.log', '.exit'):
            continue
        data = path.read_bytes()
        manifest.append({'path': str(path.relative_to(ROOT)), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
    (ROOT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'{len(rows)} timing rows from {len(files)} JSON files; manifest has {len(manifest)} artifacts')


if __name__ == '__main__':
    main()
