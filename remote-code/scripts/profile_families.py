"""Retained production-shape seven-arm profiling; supervisor launches on GPU only."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--active-seconds', type=float, default=5.)
    parser.add_argument('--campaign-start', type=float, required=True)
    parser.add_argument('--deadline', type=float, required=True)
    parser.add_argument('--gpu-csv', type=Path)
    arguments = parser.parse_args()
    if arguments.active_seconds <= 0: parser.error('Active seconds must be positive')
    root = arguments.root.resolve()
    from src.runtime import configure
    configure(root)
    from src.runner import train_one
    from src.protocol import ARMS
    from src.checkpoint import atomic_json
    from scripts.profile import summarize
    smoke_path = root / 'results/integrated_smoke.json'
    if not smoke_path.exists() or not json.loads(smoke_path.read_text()).get('passed'):
        raise RuntimeError('Integrated seven-arm GPU smoke must pass before production-shape profiling')
    report = dict(started_at=time.time(), passed=False, active_seconds=arguments.active_seconds,
                  shape={'batch_size': 256, 'block_steps': 4, 'hidden_size': 64}, runs={}, failures=[])
    destination = root / 'results/production_profile_gate.json'
    atomic_json(destination, report)
    for method in ARMS:
        if time.time() >= arguments.deadline:
            report['failures'].append({'method': method, 'reason': 'worker deadline reached before launch'})
            break
        config = json.loads((root / 'configs/methods' / (method + '.yaml')).read_text())
        config.update(method=method, run_id='profile_' + method, purpose='production_shape_profile',
                      campaign_start=arguments.campaign_start, deadline=arguments.deadline)
        if any(config.get(key) != value for key, value in report['shape'].items()):
            raise RuntimeError('Production profiling config shape differs from frozen shape: ' + method)
        print('PROFILE_START', method, flush=True)
        result = train_one(root, config, arguments.active_seconds, resume=True)
        report['runs'][method] = {key: result.get(key) for key in
            ('run_id', 'status', 'outer_rounds', 'active_training_seconds', 'compile_seconds',
             'learner_steps', 'last_checkpoint_id', 'error')}
        if result.get('status') != 'completed' or not result.get('outer_rounds'):
            report['failures'].append({'method': method, 'reason': result.get('error', result.get('status'))})
        atomic_json(destination, report)
        mapping = {'*': str(arguments.gpu_csv)} if arguments.gpu_csv else {}
        atomic_json(root / 'results/profile.json', summarize(root, mappings=mapping))
        print('PROFILE_DONE', method, result.get('status'), flush=True)
        if report['failures']:
            # Invalid profiling blocks extra compute; retained artifacts remain.
            break
    report.update(finished_at=time.time(), passed=not report['failures'] and len(report['runs']) == len(ARMS))
    report['scope'] = 'Full-shape validity/throughput observations, not convergence or scientific ranking; inspect block counts before runtime allocation'
    atomic_json(destination, report)
    print(json.dumps({'passed': report['passed'], 'failures': report['failures']}), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
